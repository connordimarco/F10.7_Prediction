#!/usr/bin/env python3
"""Assemble the deliverable PDF from the current plot PNGs."""

import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = [
    os.path.join(HERE, "plots", "data_story.png"),
    os.path.join(HERE, "F107_E09_combo", "out", "plots", "blend_search.png"),
    os.path.join(HERE, "F107_E09_combo", "out", "plots", "forecast_examples.png"),
]
imgs = [Image.open(p).convert("RGB") for p in PAGES]
out = os.path.join(HERE, "plots", "f107_prediction.pdf")
imgs[0].save(out, save_all=True, append_images=imgs[1:], resolution=150)
print("wrote", out, f"({len(imgs)} pages)")
