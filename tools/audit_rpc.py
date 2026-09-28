"""DL1: audit the raw RPC checkout data (val2019 + test2019) before anything is derived from it.

Reads the two raw annotation files and the image headers (no pixel decoding) and reports:
images/objects per source and level, objects per SKU, image sizes, which fields are present,
id/file-name overlap between val2019 and test2019, how well the image grouping of
``autocheckout.groups`` holds (IMPLEMENTATION_PLAN.md, section 4.2) and, with ``--md5``,
byte-identical image files. Writes ``<out>/audit.json`` (all numbers) and ``<out>/audit.md``.

    python -m tools.audit_rpc --raw /data/rpc/raw/retail_product_checkout --out results/data_audit --md5
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from autocheckout.groups import Multiset, group_keys, image_multisets, parse_file_name
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.rpc import LEVELS, SOURCES, raw_ann_path, raw_image_dir

MISSING = "<missing>"
# The largest group is flagged as a giant component (chained groups) above this share of images.
GIANT_GROUP_SHARE = 0.01


def histogram(values: list[int]) -> dict[str, int]:
    """``{value: count}`` with numeric keys in increasing order (as strings, for JSON)."""
    return {str(value): count for value, count in sorted(Counter(values).items())}


def count_keys(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(key for record in records for key in record).items()))


def level_order(level: str) -> tuple[int, str]:
    return (LEVELS.index(level), level) if level in LEVELS else (len(LEVELS), level)


def annotation_stats(coco: dict[str, Any]) -> dict[str, Any]:
    """Counts that only need the annotation file of one source."""
    images = coco["images"]
    anns = coco["annotations"]
    level_of = {img["id"]: str(img.get("level", MISSING)) for img in images}
    size_of = {img["id"]: (img.get("width", 0), img.get("height", 0)) for img in images}
    images_per_level = Counter(level_of.values())
    objects_per_level = Counter(level_of.get(ann["image_id"], MISSING) for ann in anns)
    bad_boxes = 0
    for ann in anns:
        x, y, w, h = ann["bbox"]
        width, height = size_of.get(ann["image_id"], (0, 0))
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            bad_boxes += 1
    image_ids_with_objects = {ann["image_id"] for ann in anns}
    return {
        "images": len(images),
        "objects": len(anns),
        "images_without_objects": sum(img["id"] not in image_ids_with_objects for img in images),
        "annotations_without_image": sum(ann["image_id"] not in level_of for ann in anns),
        "images_per_level": {k: images_per_level[k] for k in sorted(images_per_level, key=level_order)},
        "objects_per_level": {k: objects_per_level[k] for k in sorted(objects_per_level, key=level_order)},
        "json_image_sizes": dict(sorted(Counter(f"{w}x{h}" for w, h in size_of.values()).items())),
        "image_fields": count_keys(images),
        "annotation_fields": count_keys(anns),
        "category_fields": count_keys(coco["categories"]),
        "iscrowd_values": dict(sorted(Counter(str(ann.get("iscrowd", MISSING)) for ann in anns).items())),
        "boxes_empty_or_outside_image": bad_boxes,
        "duplicate_image_ids": len(images) - len(level_of),
        "duplicate_file_names": len(images) - len({img["file_name"] for img in images}),
    }


def objects_per_sku(cocos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    category_ids = sorted(c["id"] for c in cocos[SOURCES[0]]["categories"])
    per_source = {source: Counter(ann["category_id"] for ann in coco["annotations"])
                  for source, coco in cocos.items()}
    total = sum(per_source.values(), Counter())
    return {
        "per_sku": {str(cid): {source: per_source[source][cid] for source in cocos} | {"total": total[cid]}
                    for cid in category_ids},
        "min_total": min(total[cid] for cid in category_ids),
        "max_total": max(total[cid] for cid in category_ids),
        "skus_without_objects": {source: [cid for cid in category_ids if per_source[source][cid] == 0]
                                 for source in cocos},
        "unknown_category_ids": sorted(set(total) - set(category_ids)),
    }


def category_stats(cocos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    categories = cocos[SOURCES[0]]["categories"]
    per_super = Counter(str(c.get("supercategory", MISSING)) for c in categories)
    return {
        "count": len(categories),
        "identical_in_all_sources": all(coco["categories"] == categories for coco in cocos.values()),
        "id_range": [min(c["id"] for c in categories), max(c["id"] for c in categories)],
        "skus_per_supercategory": dict(sorted(per_super.items())),
    }


def overlap_stats(cocos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first, second = (cocos[source]["images"] for source in SOURCES)
    shared_ids = {img["id"] for img in first} & {img["id"] for img in second}
    shared_names = {img["file_name"] for img in first} & {img["file_name"] for img in second}
    return {
        "shared_image_ids": len(shared_ids),
        "shared_file_names": len(shared_names),
        "shared_file_name_examples": sorted(shared_names)[:20],
    }


def read_header(path: Path) -> tuple[str, str] | None:
    """(``WxH``, mode) from the image header without decoding pixels; None if the file is missing."""
    if not path.exists():
        return None
    with Image.open(path) as image:
        return f"{image.width}x{image.height}", image.mode


def file_stats(coco: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    sizes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    missing, mismatched = [], []
    for img in tqdm(coco["images"], desc=f"headers {image_dir.name}"):
        header = read_header(image_dir / img["file_name"])
        if header is None:
            missing.append(img["file_name"])
            continue
        size, mode = header
        sizes[size] += 1
        modes[mode] += 1
        if size != f"{img.get('width')}x{img.get('height')}":
            mismatched.append(img["file_name"])
    on_disk = {name for name in os.listdir(image_dir) if name.lower().endswith(".jpg")}
    return {
        "file_sizes": dict(sorted(sizes.items())),
        "file_modes": dict(sorted(modes.items())),
        "missing_files": len(missing),
        "missing_file_examples": missing[:20],
        "size_differs_from_json": len(mismatched),
        "size_differs_examples": mismatched[:20],
        "jpg_files_on_disk": len(on_disk),
        "jpg_files_not_in_json": len(on_disk - {img["file_name"] for img in coco["images"]}),
    }


Row = tuple[str, str, str, Multiset, str]  # (source, file_name, suffix, SKU multiset, level)


def image_rows(cocos: dict[str, dict[str, Any]]) -> tuple[list[Row], list[str]]:
    """One row per image of every source, plus the file names that do not parse."""
    rows, unparseable = [], []
    for source in SOURCES:
        multisets = image_multisets(cocos[source])
        for img in cocos[source]["images"]:
            try:
                suffix = parse_file_name(img["file_name"])[1]
            except ValueError:
                unparseable.append(f"{source}/{img['file_name']}")
                continue
            level = str(img.get("level", MISSING))
            rows.append((source, img["file_name"], suffix, multisets[img["id"]], level))
    return rows, unparseable


def grouping_stats(rows: list[Row], keys: list[str]) -> dict[str, Any]:
    """How well "same suffix" and "same SKU multiset" describe groups of shots of one basket."""
    by_suffix: dict[str, list[int]] = defaultdict(list)
    by_multiset: dict[Multiset, list[int]] = defaultdict(list)
    for i, (_, _, suffix, multiset, _) in enumerate(rows):
        by_suffix[suffix].append(i)
        if multiset:
            by_multiset[multiset].append(i)
    suffix_sources = {s: {rows[i][0] for i in idx} for s, idx in by_suffix.items()}
    suffixes_in_both = [s for s, sources in suffix_sources.items() if len(sources) > 1]
    multisets_per_suffix = [len({rows[i][3] for i in idx}) for idx in by_suffix.values()]
    suffixes_per_multiset = [len({rows[i][2] for i in idx}) for idx in by_multiset.values()]

    members: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        members[key].append(i)
    sizes = sorted((len(idx) for idx in members.values()), reverse=True)
    multi_suffix_groups = [idx for idx in members.values() if len({rows[i][2] for i in idx}) > 1]
    cross_source_groups = [idx for idx in members.values() if len({rows[i][0] for i in idx}) > 1]
    return {
        "images": len(rows),
        "suffixes": len(by_suffix),
        "suffixes_per_source": {source: sum(source in s for s in suffix_sources.values())
                                for source in SOURCES},
        "suffixes_in_both_sources": len(suffixes_in_both),
        "images_with_suffix_in_both_sources": sum(len(by_suffix[s]) for s in suffixes_in_both),
        "images_per_suffix": histogram([len(idx) for idx in by_suffix.values()]),
        "suffixes_with_one_multiset": sum(n == 1 for n in multisets_per_suffix),
        "distinct_multisets_per_suffix": histogram(multisets_per_suffix),
        "suffixes_spanning_levels": sum(len({rows[i][4] for i in idx}) > 1 for idx in by_suffix.values()),
        "images_without_objects": sum(not row[3] for row in rows),
        "distinct_multisets": len(by_multiset),
        "multisets_shared_by_several_suffixes": sum(n > 1 for n in suffixes_per_multiset),
        "suffixes_per_multiset": histogram(suffixes_per_multiset),
        "groups": len(members),
        "suffix_groups_merged_by_multiset": len(by_suffix) - len(members),
        "groups_with_several_suffixes": len(multi_suffix_groups),
        "images_in_groups_with_several_suffixes": sum(len(idx) for idx in multi_suffix_groups),
        "group_sizes": histogram(sizes),
        "largest_groups": sizes[:10],
        "giant_group": sizes[0] > GIANT_GROUP_SHARE * len(rows),
        "groups_spanning_levels": sum(len({rows[i][4] for i in idx}) > 1 for idx in members.values()),
        "groups_spanning_sources": len(cross_source_groups),
        "images_in_groups_spanning_sources": sum(len(idx) for idx in cross_source_groups),
    }


def md5_duplicates(paths: list[Path], workers: int, group_of: dict[str, str] | None) -> dict[str, Any]:
    """Sets of byte-identical files (``<source>/<file_name>``) and how many of them span several groups."""
    with Pool(workers) as pool:
        digests = list(tqdm(pool.imap(md5_file, paths, chunksize=64), total=len(paths), desc="md5"))
    by_digest: dict[str, list[str]] = defaultdict(list)
    for path, digest in zip(paths, digests, strict=True):
        by_digest[digest].append(f"{path.parent.name}/{path.name}")
    duplicates = sorted(sorted(names) for names in by_digest.values() if len(names) > 1)
    result = {"files_hashed": len(paths), "duplicate_sets": len(duplicates),
              "files_in_duplicate_sets": sum(len(names) for names in duplicates), "sets": duplicates}
    if group_of is not None:
        result["sets_spanning_groups"] = sum(len({group_of[name] for name in names}) > 1
                                             for names in duplicates)
    return result


def audit(raw_dir: Path, *, md5: bool, workers: int) -> dict[str, Any]:
    cocos = {source: load_json(raw_ann_path(raw_dir, source)) for source in SOURCES}
    result: dict[str, Any] = {"raw_dir": str(raw_dir), "sources": {}}
    for source in SOURCES:
        stats = annotation_stats(cocos[source])
        stats.update(file_stats(cocos[source], raw_image_dir(raw_dir, source)))
        result["sources"][source] = stats
    result["categories"] = category_stats(cocos)
    result["objects_per_sku"] = objects_per_sku(cocos)
    result["overlap"] = overlap_stats(cocos)
    rows, unparseable = image_rows(cocos)
    group_of = None
    if unparseable:
        result["grouping"] = {"error": "file names not of the form YYYYMMDD-HH-MM-SS-<suffix>.jpg; "
                                       "grouping not computed",
                              "unparseable_file_names": len(unparseable), "examples": unparseable[:20]}
    else:
        keys = group_keys([row[1] for row in rows], [row[3] for row in rows])
        result["grouping"] = grouping_stats(rows, keys)
        group_of = {f"{row[0]}/{row[1]}": key for row, key in zip(rows, keys, strict=True)}
    if md5:
        paths = [raw_image_dir(raw_dir, source) / img["file_name"]
                 for source in SOURCES for img in cocos[source]["images"]]
        result["md5"] = md5_duplicates([p for p in paths if p.exists()], workers, group_of)
    return result


def render_markdown(result: dict[str, Any]) -> str:
    sources = result["sources"]
    lines = ["# RPC data audit (DL1)", "",
             "Generated by `python -m tools.audit_rpc`; every number is in `audit.json`.", "",
             "## Images and objects", "",
             "| source | images | objects | images per level | objects per level |",
             "|---|---|---|---|---|"]
    for source, s in sources.items():
        lines.append(f"| {source} | {s['images']} | {s['objects']} | {s['images_per_level']} | "
                     f"{s['objects_per_level']} |")
    per_sku = result["objects_per_sku"]
    cats = result["categories"]
    lines += ["", f"- Categories: {cats['count']} (ids {cats['id_range'][0]}..{cats['id_range'][1]}), "
              f"{len(cats['skus_per_supercategory'])} supercategories; identical in both files: "
              f"{cats['identical_in_all_sources']}.",
              f"- Objects per SKU (both sources): min {per_sku['min_total']}, max {per_sku['max_total']}; "
              f"SKUs without objects: {per_sku['skus_without_objects']}; "
              f"unknown category ids: {per_sku['unknown_category_ids']}.",
              "", "## Fields and sizes", ""]
    for source, s in sources.items():
        n_img, n_ann = s["images"], s["objects"]
        image_fields, ann_fields = s["image_fields"], s["annotation_fields"]
        lines.append(f"- {source}: images with `level` {image_fields.get('level', 0)}/{n_img}; "
                     f"annotations with `area` {ann_fields.get('area', 0)}/{n_ann}, "
                     f"`iscrowd` {ann_fields.get('iscrowd', 0)}/{n_ann} (values {s['iscrowd_values']}); "
                     f"categories with `supercategory` {s['category_fields'].get('supercategory', 0)}"
                     f"/{cats['count']}.")
        lines.append(f"  Sizes in JSON {s['json_image_sizes']}, in files {s['file_sizes']} "
                     f"(modes {s['file_modes']}); size differs from JSON: {s['size_differs_from_json']}; "
                     f"missing files: {s['missing_files']}; "
                     f"jpg on disk not in JSON: {s['jpg_files_not_in_json']}; "
                     f"boxes empty or outside the image: {s['boxes_empty_or_outside_image']}.")
    overlap = result["overlap"]
    lines += ["", "## val2019 / test2019 overlap", "",
              f"- Shared image ids: {overlap['shared_image_ids']}; "
              f"shared file names: {overlap['shared_file_names']}.",
              "", "## Groups (plan 4.2)", ""]
    g = result["grouping"]
    if "error" in g:
        lines.append(f"- **{g['error']}** ({g['unparseable_file_names']} names, e.g. {g['examples'][:3]}).")
    else:
        lines += [
            f"- Suffixes: {g['suffixes']} (per source {g['suffixes_per_source']}); in both sources: "
            f"{g['suffixes_in_both_sources']} ({g['images_with_suffix_in_both_sources']} images).",
            f"- Images per suffix: {g['images_per_suffix']}.",
            f"- Suffixes whose images all share one SKU multiset: "
            f"{g['suffixes_with_one_multiset']}/{g['suffixes']} "
            f"(distinct multisets per suffix: {g['distinct_multisets_per_suffix']}); "
            f"suffixes spanning levels: {g['suffixes_spanning_levels']}.",
            f"- Multisets shared by several suffixes: {g['multisets_shared_by_several_suffixes']}; "
            f"suffix groups merged by identical multisets: {g['suffix_groups_merged_by_multiset']} "
            f"({g['images_in_groups_with_several_suffixes']} images end up in groups with several suffixes).",
            f"- Final groups: {g['groups']}; sizes {g['group_sizes']}; largest {g['largest_groups']}; "
            f"giant group (> {GIANT_GROUP_SHARE:.0%} of images): **{g['giant_group']}**.",
            f"- Groups spanning levels: {g['groups_spanning_levels']}; spanning sources: "
            f"{g['groups_spanning_sources']} ({g['images_in_groups_spanning_sources']} images, "
            "all go to train).",
        ]
    if "md5" in result:
        m = result["md5"]
        lines += ["", "## Byte-identical images", "",
                  f"- {m['files_hashed']} files hashed; {m['duplicate_sets']} duplicate sets "
                  f"({m['files_in_duplicate_sets']} files), spanning several groups: "
                  f"{m.get('sets_spanning_groups', 'not computed')}; first sets (up to 3 files each): "
                  f"{[names[:3] for names in m['sets'][:3]]}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="folder with val2019/, test2019/, the JSONs")
    parser.add_argument("--out", type=Path, default=Path("results/data_audit"))
    parser.add_argument("--md5", action="store_true", help="also hash every image file to find duplicates")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)

    result = audit(args.raw, md5=args.md5, workers=args.workers)
    save_json(args.out / "audit.json", result, indent=1)
    markdown = render_markdown(result)
    (args.out / "audit.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
