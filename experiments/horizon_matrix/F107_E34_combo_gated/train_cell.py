#!/usr/bin/env python3
"""E34: run all 30 leads of e34_lead.py sequentially (local), then merge.
On Athena use the lead fan-out instead (run_e30.pbs, same lead files)."""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e34_lead

for h in range(1, 31):
    e34_lead.lead(h)
subprocess.check_call([sys.executable, os.path.join(HERE, "merge_leads.py")])
