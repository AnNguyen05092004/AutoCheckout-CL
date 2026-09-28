"""Groups of RPC checkout images that show the same basket (IMPLEMENTATION_PLAN.md, section 4.2).

RPC checkout images are named ``YYYYMMDD-HH-MM-SS-<suffix>.jpg``. Images come in runs of
consecutive shots with the same suffix (the same basket photographed several times), and a
suffix can reappear at other times. Splitting image by image would put near-identical shots in
train and test, so splits are made per group:

- images with the same suffix are in the same group (pooled across val2019 and test2019);
- images with the same non-empty SKU multiset ``{(category_id, count)}`` are in the same group;
- groups are the connected components of these two relations (union-find).

The key of a group is ``"g" + <smallest suffix in the group>`` (numeric order); it does not
depend on the order of the input images.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

FILE_NAME_RE = re.compile(r"^(\d{8}-\d{2}-\d{2}-\d{2})-(\d+)\.jpg$")

Multiset = tuple[tuple[int, int], ...]


def parse_file_name(file_name: str) -> tuple[str, str]:
    """Split ``20180827-13-42-20-204.jpg`` into (timestamp ``20180827-13-42-20``, suffix ``204``)."""
    match = FILE_NAME_RE.match(file_name)
    if match is None:
        raise ValueError(f"file name {file_name!r} does not match YYYYMMDD-HH-MM-SS-<suffix>.jpg")
    return match.group(1), match.group(2)


def sku_multiset(category_ids: Iterable[int]) -> Multiset:
    """Sorted ``((category_id, count), ...)`` of the objects in one image."""
    return tuple(sorted(Counter(category_ids).items()))


def image_multisets(coco: dict[str, Any]) -> dict[int, Multiset]:
    """SKU multiset of every image of a COCO dict (images without objects get ``()``)."""
    categories: dict[int, list[int]] = defaultdict(list)
    for ann in coco["annotations"]:
        categories[ann["image_id"]].append(ann["category_id"])
    return {img["id"]: sku_multiset(categories[img["id"]]) for img in coco["images"]}


def _suffix_order(suffix: str) -> tuple[int, str]:
    return len(suffix), suffix


def group_keys(file_names: Sequence[str], multisets: Sequence[Multiset]) -> list[str]:
    """Group key of each image (same position as the inputs)."""
    parent = list(range(len(file_names)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    suffixes = [parse_file_name(name)[1] for name in file_names]
    first_by_suffix: dict[str, int] = {}
    first_by_multiset: dict[Multiset, int] = {}
    for i, (suffix, multiset) in enumerate(zip(suffixes, multisets, strict=True)):
        union(i, first_by_suffix.setdefault(suffix, i))
        if multiset:  # never merge images just because both are empty
            union(i, first_by_multiset.setdefault(multiset, i))

    smallest: dict[int, str] = {}
    for i, suffix in enumerate(suffixes):
        root = find(i)
        if root not in smallest or _suffix_order(suffix) < _suffix_order(smallest[root]):
            smallest[root] = suffix
    return ["g" + smallest[find(i)] for i in range(len(file_names))]


def coco_group_keys(coco: dict[str, Any]) -> dict[int, str]:
    """Group key of every image of a COCO dict whose ``file_name``s are original RPC names.

    Uses ``orig_file_name`` when present (merged annotation, where ``file_name`` may carry a
    source prefix).
    """
    multisets = image_multisets(coco)
    images = coco["images"]
    names = [img.get("orig_file_name", img["file_name"]) for img in images]
    keys = group_keys(names, [multisets[img["id"]] for img in images])
    return {img["id"]: key for img, key in zip(images, keys, strict=True)}
