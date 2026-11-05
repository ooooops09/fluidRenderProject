#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fluidrender.core import load_config
from fluidrender.practical import run_crypto,run_serving
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['crypto','serving'],default='crypto')
p.add_argument('--config',default=str(ROOT/'configs/default.json'))
p.add_argument('--checkpoint')
p.add_argument('--quick',action='store_true')
p.add_argument('--samples',type=int,default=120)
p.add_argument('--out',default=str(ROOT/'results/overhead'))
a=p.parse_args()
if a.mode=='serving' and not a.checkpoint:p.error('--checkpoint is required for serving')
print(json.dumps(run_crypto(a.out,a.quick) if a.mode=='crypto' else run_serving(load_config(a.config),a.checkpoint,a.out,a.samples),indent=2))
