#!/usr/bin/env python3
"""E60: E21 (robust flux, level target, train 1996-2021) + the Bremen Mg II
plage-proxy index as an optional-NaN 60-day window (60 extra features)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import single_model

single_model.run(HERE, flux="f107_adj_rob", target="raw", extra=["mgii"])
