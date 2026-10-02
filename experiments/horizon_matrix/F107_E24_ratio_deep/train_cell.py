#!/usr/bin/env python3
"""E24: E22 trained on the 1947-2021 span (ar_*/fs_* optional-NaN) — three more maxima for the ratio model."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import single_model

single_model.run(HERE, flux="f107_adj_rob", target="ratio_env", span="train47")
