"""Export per-bearing seed tables and independent bar charts from a metrics CSV."""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('csv',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
df=pd.read_csv(a.csv)
required={'target','arm','seed','rmse_normalized'}
if not required.issubset(df.columns):raise ValueError(f'Missing columns: {required-set(df.columns)}')
if df.duplicated(['target','arm','seed']).any():raise ValueError('Duplicate seed rows')
a.output.mkdir(parents=True,exist_ok=False)
df.to_csv(a.output/'per_seed.csv',index=False)
stat=df.groupby(['target','arm'],sort=False).rmse_normalized.agg(['count','mean','std'])
stat.to_csv(a.output/'statistics.csv')
pivot=df.pivot(index=['target','seed'],columns='arm',values='rmse_normalized')
pivot.to_csv(a.output/'comparison.csv')
for target,sub in stat.groupby(level=0,sort=False):
    rows=sub.droplevel(0)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(rows.index,rows['mean'],yerr=rows['std'].fillna(0),capsize=5)
    ax.set_ylabel('RMSE / max(true RUL)');ax.set_title(target)
    ax.tick_params(axis='x',labelsize=9)
    fig.tight_layout();fig.savefig(a.output/f'{target}.png',dpi=180);plt.close(fig)
print(stat.to_string())
