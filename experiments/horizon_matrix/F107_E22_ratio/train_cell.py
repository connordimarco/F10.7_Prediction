#!/usr/bin/env python3
"""E22: E21 with the target expressed as a ratio to the 81-day envelope at the origin (level factored out; trees no longer cap the tail)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import single_model

single_model.run(HERE, flux="f107_adj_rob", target="ratio_env")
