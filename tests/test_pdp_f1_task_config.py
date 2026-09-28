"""F1: task setup for RPC comes from a task config file instead of the hard-coded COCO split."""

from datasets.create_coco_instance import task_info_rpc
from pdp_helpers import write_task_config


def test_task_map_matches_config_and_covers_all_slots(tmp_path):
    path = write_task_config(tmp_path / "tasks.json")
    task_map, label2name = task_info_rpc(str(path))

    assert sorted(task_map) == [1, 2, 3, 4, 5, 6]
    sizes = [task_map[t][2] for t in sorted(task_map)]
    assert sizes == [100, 25, 25, 25, 25, 24]
    assert sum(sizes) == 224
    # offsets are contiguous: each task starts where the previous one ends
    offsets = [task_map[t][1] for t in sorted(task_map)]
    assert offsets == [0, 100, 125, 150, 175, 200]
    assert task_map[2][0][0] == "sku_101" and label2name[100] == "sku_101"
    assert label2name[223] == "reserved_223" and len(label2name) == 224
