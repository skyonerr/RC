"""Separate prediction figures for each bearing and arm."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(out):
    for target in sorted(out.glob('Bearing*')):
        for arm in sorted(p for p in target.iterdir() if p.is_dir()):
            fig,ax=plt.subplots(figsize=(8,4))
            found=False
            for folder in sorted(arm.glob('seed*')):
                p=folder/'predictions.csv'
                if not p.exists():continue
                df=pd.read_csv(p)
                if not found:ax.plot(df.True_RUL,color='black',label='True RUL',linewidth=2)
                ax.plot(df.Pred_RUL,label=folder.name,alpha=.75)
                found=True
            if found:
                ax.set(title=f'{target.name}: {arm.name}',xlabel='Window index',ylabel='RUL (original label scale)')
                ax.legend();fig.tight_layout();fig.savefig(arm/'predictions.png',dpi=160)
            plt.close(fig)
