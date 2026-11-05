#!/usr/bin/env python3
"""Entry point for cache-freshness and recovery replay."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('run_replay.py')),run_name='__main__')
