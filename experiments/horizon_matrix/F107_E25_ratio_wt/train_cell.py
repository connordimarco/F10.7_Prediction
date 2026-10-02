#!/usr/bin/env python3
"""E25: E22 with sample weights proportional to the origin envelope (high-activity origins count up to 3x)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import single_model

single_model.run(HERE, flux="f107_adj_rob", target="ratio_env", weights="env")
