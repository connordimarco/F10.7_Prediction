#!/usr/bin/env python3
"""E20: v2-data control — E10 features, cfg-44 params, canonical flux, raw target (single model, train 1996-2021)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import single_model

single_model.run(HERE, flux="f107_adj", target="raw")
