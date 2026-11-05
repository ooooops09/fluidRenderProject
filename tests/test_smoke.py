"""Short-run integration test for heuristic schedulers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fluidrender.experiments import load_and_override,run_experiment

def test_short_run(tmp_path):
    cfg=load_and_override(str(ROOT/'configs/default.json'),2,0,20)
    summary=run_experiment(cfg,['RR','SAG','A-LALL'],1,tmp_path)
    for row in summary.values():
        assert row['tasks']>0
        assert 0 <= row['jain_util'] <= 1
