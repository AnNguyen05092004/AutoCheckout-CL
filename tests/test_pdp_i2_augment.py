"""I2: box coordinates follow the image under rotations and flips; annotations are never mutated."""

import copy
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from augment import TrainAugment, hflip, rotate90
from datasets.coco_hug import CocoDetection
from pdp_helpers import small_processor

BOX = [7, 5, 20, 9]  # x, y, w, h on a 60 x 40 image


def red_image():
    image = Image.new("RGB", (60, 40), (255, 255, 255))
    x, y, w, h = BOX
    ImageDraw.Draw(image).rectangle([x, y, x + w - 1, y + h - 1], fill=(255, 0, 0))
    return image


def red_box(image):
    red = np.asarray(image)[..., 1] < 128
    ys, xs = np.nonzero(red)
    return [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_rotation_moves_boxes_with_the_pixels(k):
    anns = [{"bbox": list(BOX), "category_id": 1, "area": 180}]
    original = copy.deepcopy(anns)
    image, out = rotate90(red_image(), anns, k)
    assert anns == original  # inputs untouched
    np.testing.assert_allclose(out[0]["bbox"], red_box(image))
    assert out[0]["area"] == 180 and image.size == ((60, 40) if k % 2 == 0 else (40, 60))


def test_flip_moves_boxes_with_the_pixels():
    image, out = hflip(red_image(), [{"bbox": list(BOX)}])
    np.testing.assert_allclose(out[0]["bbox"], red_box(image))


def test_dataset_applies_augmentation_without_touching_the_coco_index(tmp_path):
    red_image().save(tmp_path / "a.jpg")
    ann = {"images": [{"id": 1, "file_name": "a.jpg", "width": 60, "height": 40}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": list(BOX), "area": 180, "iscrowd": 0}],
           "categories": [{"id": 0, "name": "x"}]}
    (tmp_path / "a.json").write_text(json.dumps(ann))
    dataset = CocoDetection(str(tmp_path), str(tmp_path / "a.json"), small_processor(),
                            augment=TrainAugment(scales=(48,), color=0))
    before = copy.deepcopy(dataset.coco.anns)
    for _ in range(4):
        pixels, target = dataset[0]
        assert target["boxes"].min() >= 0 and target["boxes"].max() <= 1
        assert min(pixels.shape[1:]) == 48  # shortest edge from the augmentation
    assert dataset.coco.anns == before
