"""Copy a folder of per-task COCO files keeping only the first N images of each file (R4 GPU smoke).

    python -m tools.subset_tasks --src /data/rpc/tasks/pilot_100-4x25_seed0 --dst /data/rpc/tasks/smoke --images 100

Images are taken in id order; annotations follow their images; categories are kept as they are.
manifest.json is not copied (the subset is not a data release, only a quick end-to-end check).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from autocheckout.io import load_json, save_json


def subset(coco: dict[str, Any], n_images: int) -> dict[str, Any]:
    keep = {img["id"] for img in sorted(coco["images"], key=lambda img: img["id"])[:n_images]}
    return {**coco, "images": [img for img in coco["images"] if img["id"] in keep],
            "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in keep]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--images", type=int, default=100)
    args = parser.parse_args(argv)
    for path in sorted(args.src.glob("*.json")):
        if path.name != "manifest.json":
            data = subset(load_json(path), args.images)
            save_json(args.dst / path.name, data)
            print(f"{path.name:32s} images {len(data['images']):5d}  objects {len(data['annotations']):6d}")


if __name__ == "__main__":
    main()
