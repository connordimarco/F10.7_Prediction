#!/usr/bin/env python3
"""E61: E24 (robust flux, envelope-ratio target, train 1947-2021) + the
Bremen Mg II index as an optional-NaN window (NaN before 1978-11)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import single_model

single_model.run(HERE, flux="f107_adj_rob", target="ratio_env", span="train47", extra=["mgii"])
