"""Training augmentation (improvement I2); the original code has none.

Applied to the PIL image and its COCO annotations before the image processor:
- rotation by a random multiple of 90 degrees (products lie in any orientation on the counter);
- mild colour jitter;
- a random shortest edge, returned to the caller to resize with the processor (multi-scale).
No flips by default: a mirrored package, with mirrored text and logo, never appears on a real counter,
and many RPC SKUs differ only in their printed text. Flipping both axes is a 180-degree rotation anyway.
"""

import random

from PIL import Image
from torchvision.transforms import ColorJitter

SCALES = (640, 672, 704, 736, 768, 800)


def rotate90(image, annotations, k):
    """Rotate counter-clockwise by k * 90 degrees; returns new annotation dicts (inputs untouched)."""
    width, height = image.size
    k %= 4
    out = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        if k == 1:
            box = [y, width - x - w, h, w]
        elif k == 2:
            box = [width - x - w, height - y - h, w, h]
        elif k == 3:
            box = [height - y - h, x, h, w]
        else:
            box = [x, y, w, h]
        out.append({**ann, 'bbox': box})
    transpose = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}.get(k)
    return (image.transpose(transpose) if transpose is not None else image), out


def hflip(image, annotations):
    width = image.size[0]
    out = [{**ann, 'bbox': [width - ann['bbox'][0] - ann['bbox'][2], *ann['bbox'][1:]]} for ann in annotations]
    return image.transpose(Image.FLIP_LEFT_RIGHT), out


class TrainAugment:
    def __init__(self, scales=SCALES, flip=False, color=0.2):
        self.scales = scales
        self.flip = flip
        self.jitter = ColorJitter(brightness=color, contrast=color, saturation=color, hue=0.02) if color else None

    def __call__(self, image, annotations):
        """Returns (image, new annotation dicts, processor size) for one training sample."""
        image, annotations = rotate90(image, annotations, random.randrange(4))
        if self.flip and random.random() < 0.5:
            image, annotations = hflip(image, annotations)
        if self.jitter is not None:
            image = self.jitter(image)
        return image, annotations, {'shortest_edge': random.choice(self.scales), 'longest_edge': 1333}
