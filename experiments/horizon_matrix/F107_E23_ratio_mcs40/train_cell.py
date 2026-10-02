#!/usr/bin/env python3
"""E23: E22 with min_child_samples 40 instead of 160 (does the ratio target want less regularization?)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import single_model

single_model.run(HERE, flux="f107_adj_rob", target="ratio_env", params={"min_child_samples": 40})
