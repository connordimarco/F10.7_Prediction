#!/usr/bin/env python3
"""E26: E21 with the target as a ratio to the origin-day flux (persistence-relative) instead of the envelope."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import single_model

single_model.run(HERE, flux="f107_adj_rob", target="ratio_pers")
