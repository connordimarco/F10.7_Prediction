#!/usr/bin/env python3
"""E40: GRU sequence model (shared/nn_model.py), joint 30-lead output in
envelope-ratio space, flux-space MSE loss, 5-seed ensemble, train 1947-2021.
The first non-tree family in the matrix."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import nn_model

nn_model.run(HERE, kind="gru")
