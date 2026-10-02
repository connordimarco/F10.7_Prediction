#!/usr/bin/env python3
"""E21: E20 with the flare-robust flux series (f107_adj_rob) as feature block + target."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import single_model

single_model.run(HERE, flux="f107_adj_rob", target="raw")
