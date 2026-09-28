import pytest

from autocheckout.taskcfg import TaskConfig


def make_config_dict(sizes=(3, 2), reserved=1):
    tasks, offset, rpc_id = [], 0, 1
    for task_id, size in enumerate(sizes, start=1):
        classes = []
        for i in range(size):
            classes.append({"label": offset + i, "name": f"sku{rpc_id}", "rpc_category_id": rpc_id,
                            "supercategory": "drink"})
            rpc_id += 1
        tasks.append({"task_id": task_id, "offset": offset, "reserved": False, "classes": classes})
        offset += size
    if reserved:
        tasks.append({"task_id": len(sizes) + 1, "offset": offset, "reserved": True, "classes": [
            {"label": offset + i, "name": f"reserved_{offset + i}", "rpc_category_id": None} for i in range(reserved)
        ]})
    return {"name": "toy", "seed": 0, "tasks": tasks}


def test_sizes_and_lookups():
    cfg = TaskConfig.from_dict(make_config_dict())
    assert cfg.num_slots == 6 and cfg.num_classes == 7
    assert [t.task_id for t in cfg.data_tasks] == [1, 2]
    assert cfg.seen_classes(1) == 3 and cfg.seen_classes(2) == 5
    assert cfg.task_of_label(4) == 2 and cfg.task_of_label(5) == 3
    assert cfg.rpc_to_label() == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert cfg.label_to_rpc()[3] == 4
    assert cfg.label_names()[5] == "reserved_5"


def test_round_trip(tmp_path):
    cfg = TaskConfig.from_dict(make_config_dict())
    path = tmp_path / "tasks.json"
    cfg.save(path)
    assert TaskConfig.load(path) == cfg


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["tasks"][1].update(offset=4), "offset"),
    (lambda d: d["tasks"][0]["classes"][1].update(label=7), "label"),
    (lambda d: d["tasks"][1]["classes"][0].update(rpc_category_id=1), "assigned twice"),
    (lambda d: d["tasks"][2]["classes"][0].update(rpc_category_id=99), "RPC id iff"),
    (lambda d: d["tasks"].insert(0, d["tasks"].pop()), "task ids"),
])
def test_invalid_configs_are_rejected(mutate, message):
    data = make_config_dict()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        TaskConfig.from_dict(data)
