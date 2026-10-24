"""Deterministic class colours and readable labels, drawn after image resizing."""

import colorsys
import hashlib
import os
from pathlib import Path

from PIL import ImageDraw, ImageFont

from .detection_types import COCO_CLASSES


def class_colour(label):
    common = {
        "person": (239, 68, 68), "car": (0, 190, 255), "truck": (175, 100, 255),
        "bus": (180, 130, 0), "train": (155, 100, 30), "bicycle": (255, 190, 0),
        "motorcycle": (255, 100, 190), "chair": (70, 140, 255),
        "potted plant": (60, 210, 110), "vase": (255, 130, 40),
        "bed": (210, 100, 255), "dining table": (50, 220, 220),
        "refrigerator": (120, 170, 255), "couch": (160, 80, 220), "bench": (255, 90, 120),
    }
    if label in common:
        return common[label]
    # Fixed COCO palette and deterministic name hash for unknown labels.
    seed = int.from_bytes(hashlib.sha256(label.encode("utf8")).digest()[:4], "big")
    hue = (COCO_CLASSES.index(label)*0.618033988749895)%1 if label in COCO_CLASSES else seed/2**32
    return tuple(int(v*255) for v in colorsys.hsv_to_rgb(hue, 0.78, 0.98))


def load_font(size):
    candidates = ["DejaVuSans.ttf", "Arial.ttf",
                  str(Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts/arial.ttf"),
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def draw_detections(image, detections):
    width, height = image.size
    size = max(18, min(42, round(width/48)))
    font, pad = load_font(size), max(3, size//6)
    draw = ImageDraw.Draw(image)
    occupied = []
    for item in sorted(detections, key=lambda d: -d["confidence"]):
        x1, y1, x2, y2 = [round(v*(width if i%2 == 0 else height)) for i, v in enumerate(item["bbox_norm"])]
        x1, x2 = max(0, min(width-1, x1)), max(0, min(width-1, x2))
        y1, y2 = max(0, min(height-1, y1)), max(0, min(height-1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        colour = class_colour(item["label"])
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=max(2, size//8))
        text = "{} {:.2f}".format(item["label"], item["confidence"])
        while draw.textbbox((0, 0), text, font=font)[2]+2*pad > width and len(text) > 4:
            text = text[:-2]+"…"
        bounds = draw.textbbox((0, 0), text, font=font)
        tw, th = min(width, bounds[2]-bounds[0]+2*pad), min(height, bounds[3]-bounds[1]+2*pad)
        tx = max(0, min(width-tw, x1))
        choices = [y1-th, y1, y2, y2-th]
        ty = max(0, min(height-th, y1-th))
        for candidate in choices:
            candidate = max(0, min(height-th, candidate))
            rect = (tx, candidate, tx+tw, candidate+th)
            if not any(min(rect[2], old[2]) > max(rect[0], old[0]) and min(rect[3], old[3]) > max(rect[1], old[1]) for old in occupied):
                ty = candidate
                break
        rect = (tx, ty, tx+tw, ty+th)
        draw.rectangle(rect, fill=colour)
        luminance = 0.2126*colour[0]+0.7152*colour[1]+0.0722*colour[2]
        ink = "black" if luminance > 140 else "white"
        draw.text((tx+pad-bounds[0], ty+pad-bounds[1]), text, font=font, fill=ink)
        occupied.append(rect)
    return image
