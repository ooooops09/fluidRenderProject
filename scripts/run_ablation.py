#!/usr/bin/env python3
"""Ablation policies are trained networks. This entry forwards to matched tuning."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('run_matched_tuning.py')),run_name='__main__')
