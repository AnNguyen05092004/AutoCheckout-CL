from collections import Counter

import pytest

from autocheckout.io import save_json
from autocheckout.taskcfg import TaskConfig
from tools.make_task_config import allocate, build_task_config, distribution_table, main

# 200 synthetic SKUs in 17 supercategories of uneven size, like RPC.
GROUP_SIZES = [22, 18, 16, 15, 14, 13, 12, 12, 11, 11, 10, 10, 9, 8, 7, 7, 5]
SIZES = [100, 25, 25, 25, 25]


def rpc_like_categories():
    categories = []
    for index, count in enumerate(GROUP_SIZES):
        for _ in range(count):
            cid = len(categories) + 1
            categories.append({"id": cid, "name": f"{cid}_g{index:02d}", "supercategory": f"g{index:02d}"})
    return categories


def test_allocate_small_examples():
    assert allocate({"a": 4, "b": 2}, [3, 3]) == {"a": [2, 2], "b": [1, 1]}
    # Equal remainders go to the supercategory whose name sorts first.
    assert allocate({"c": 1, "b": 1, "a": 1}, [2, 1]) == {"c": [0, 1], "b": [1, 0], "a": [1, 0]}
    with pytest.raises(ValueError, match="add up"):
        allocate({"a": 3}, [2, 2])


def test_task_sizes_are_exact_and_supercategories_proportional():
    categories = rpc_like_categories()
    config = build_task_config(categories, SIZES, reserved=24, seed=0, name="100-4x25_seed0")
    assert [t.num_classes for t in config.tasks] == [*SIZES, 24]
    assert config.num_slots == 224 and config.num_classes == 225
    assert sorted(config.rpc_to_label()) == [c["id"] for c in categories]
    per_group = Counter(c["supercategory"] for c in categories)
    for task, size in zip(config.data_tasks, SIZES, strict=True):
        counts = Counter(c.supercategory for c in task.classes)
        for group, total in per_group.items():
            assert abs(counts[group] - total * size / 200) < 1  # floor or ceiling of the exact share
        rpc_ids = [c.rpc_category_id for c in task.classes]
        assert rpc_ids == sorted(rpc_ids)
    reserved = config.tasks[-1]
    assert reserved.reserved and reserved.offset == 200
    assert [c.name for c in reserved.classes] == [f"reserved_{label}" for label in range(200, 224)]
    assert all(c.rpc_category_id is None for c in reserved.classes)


def test_seed_changes_members_but_not_counts():
    categories = rpc_like_categories()
    first = build_task_config(categories, SIZES, 24, 0, "a")
    assert first == build_task_config(categories, SIZES, 24, 0, "a")
    other = build_task_config(categories, SIZES, 24, 1, "a")
    assert other.rpc_to_label() != first.rpc_to_label()
    assert distribution_table(other) == distribution_table(first)


def test_cli_writes_a_loadable_config(tmp_path, capsys):
    save_json(tmp_path / "instances.json", {"categories": rpc_like_categories()})
    out = tmp_path / "tasks.json"
    main(["--categories", str(tmp_path / "instances.json"), "--name", "100-4x25_seed0", "--out", str(out)])
    loaded = TaskConfig.load(out)
    assert loaded == build_task_config(rpc_like_categories(), SIZES, 24, 0, "100-4x25_seed0")
    printed = capsys.readouterr().out
    assert "total                      100    25    25    25    25    200" in printed
    assert "224 slots" in printed
