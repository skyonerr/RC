"""Generate one figure per bearing and model from stored seed-0 predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    current=ROOT/'results/current'
    out=current/'figures';out.mkdir(exist_ok=True)
    arms=[('multimodal','Temperature-conditioned','#d62728'),('vibration_only','Vibration baseline','#2878b5'),('temperature_only','Temperature-only','#e69f00')]
    for target in sorted((current/'predictions').glob('Bearing*')):
        ref=None
        folder=out/target.name;folder.mkdir(exist_ok=True)
        for arm,label,color in arms:
            d=pd.read_csv(target/arm/'seed0.csv');y=d.True_RUL.to_numpy();p=d.Pred_RUL.to_numpy()
            if ref is None:
                ref=y
            else:np.testing.assert_allclose(y,ref)
            fig,ax=plt.subplots(figsize=(9,4.8))
            ax.plot(y,color='black',linewidth=2,label='True RUL')
            rmse=np.sqrt(np.mean((p-y)**2))/y.max()
            ax.plot(p,color=color,linewidth=1.3,alpha=.9,label=f'{label} (RMSE/max={rmse:.4f})')
            ax.set(title=f'{target.name} | {label} | seed 0 | 30,000 epochs',xlabel='Window index',ylabel='RUL (original label scale)')
            ax.set_ylim(-5,105)
            ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout()
            fig.savefig(folder/f'{arm}.png',dpi=180);plt.close(fig)
    print('Saved 12 separate model figures:',out)

if __name__=='__main__':main()
