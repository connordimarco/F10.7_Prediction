#!/usr/bin/env python3
"""E41: E40 with an MLP encoder on the flattened window instead of the GRU
(same inputs, loss, ensemble, span)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import nn_model

nn_model.run(HERE, kind="mlp")
