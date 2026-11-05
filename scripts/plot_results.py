#!/usr/bin/env python3
"""Plot experiment statistics from CSV files."""
import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('csv_file')
p.add_argument('--x',default='scenario')
p.add_argument('--y',default='p99_latency_ms')
p.add_argument('--error',default='p99_latency_ms_sd',help='SD/CI column; label it correctly in --ylabel')
p.add_argument('--ylabel',default='Completed-request P99 latency (ms), mean ± SD')
p.add_argument('--out',default='figure.pdf')
a=p.parse_args()
try:
    import matplotlib.pyplot as plt
except ImportError:
    p.exit(1,'Install plotting support with: python -m pip install ".[plots]"\n')
rows=list(csv.DictReader(open(a.csv_file)))
rows=[r for r in rows if r.get(a.y) not in ('',None)]
if not rows:p.error('No numerical observations in the requested column')
plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(max(5,len(rows)*.55),3.3),constrained_layout=True)
x=list(range(len(rows)));y=[float(r[a.y]) for r in rows]
err=[float(r.get(a.error) or 0) for r in rows]
ax.errorbar(x,y,yerr=err,fmt='o-',capsize=3,lw=1.4)
ax.set_xticks(x)
ax.set_xticklabels([r.get(a.x,str(i)) for i,r in enumerate(rows)],rotation=30,ha='right')
ax.set_ylabel(a.ylabel);ax.grid(axis='y',alpha=.25)
Path(a.out).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,bbox_inches='tight')
