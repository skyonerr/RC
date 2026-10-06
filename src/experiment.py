"""Prepare, train, and evaluate the configured cross-bearing RUL models."""
import json
import pickle
import random
import hashlib
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models.MLP import MLPReadout
from models.reservoir import Reservoir
from .data import get_global_stats, load_data
from .metrics import metrics

ROOT = Path(__file__).resolve().parents[1]

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def read(path):
    with path.open('rb') as stream:
        return pickle.load(stream)

def temperature_windows(payload, mean, std, cfg):
    """Construct standardized temperature-only input windows."""
    t = np.asarray(payload['train']['T_K'], dtype=float).reshape(-1)
    z = (np.clip(t, *cfg['temperature_clip_K'])-mean)/std
    return np.lib.stride_tricks.sliding_window_view(z, cfg['window']).copy()[..., None]

def prepare(target, arm, cfg):
    """Prepare balanced source windows and data from the held-out bearing."""
    case = cfg['targets'][target]
    train = [ROOT/'data/processed'/p for p in case['train']]
    test = ROOT/'data/processed'/case['test']
    assert len(set(train)) == 2 and test not in train
    if arm == 'temperature_only':
        raw = [read(p) for p in train]
        tt = np.concatenate([np.clip(np.asarray(d['train']['T_K'], dtype=float).reshape(-1), *cfg['temperature_clip_K']) for d in raw])
        mean, std = float(tt.mean()), float(tt.std())+1e-9
        parts = [(temperature_windows(d, mean, std, cfg), None, np.asarray(d['train']['y_10s'], dtype=float)[cfg['window']-1:]) for d in raw]
        raw_test = read(test)
        test_data = (temperature_windows(raw_test, mean, std, cfg), None, np.asarray(raw_test['train']['y_10s'], dtype=float)[cfg['window']-1:])
    else:
        mean, std = get_global_stats(train)
        parts = [load_data(p, mean, std, cfg['window']) for p in train]
        test_data = load_data(test, mean, std, cfg['window'])
    n = min(len(v[2]) for v in parts)
    indices = [np.linspace(0, len(y)-1, n).round().astype(int) for x,t,y in parts]
    balanced = [(x[i], None if t is None else t[i], y[i]) for (x,t,y),i in zip(parts,indices)]
    provenance = dict(train=case['train'],test=case['test'],training_window_indices=[v.tolist() for v in indices],
                      mean=np.asarray(mean).tolist(),std=np.asarray(std).tolist(),test_windows=len(test_data[2]),
                      train_windows_each=n,validation=None)
    return balanced, test_data, provenance

def run_one(target, arm, seed, cfg, folder):
    """Train one seeded model and save its predictions and configuration."""
    seed_all(seed)
    parts, (xt,tt,yt), provenance = prepare(target,arm,cfg)
    ys = np.concatenate([y for x,t,y in parts]); limit = float(ys.max())
    thermal = arm == 'multimodal'
    temp = np.concatenate([np.clip(t,*cfg['temperature_clip_K']) for x,t,y in parts]) if thermal else 375.15
    res = Reservoir(temperature=temp,T_ref=375.15,Ea=.7 if thermal else 0.,
                    fixed_lambda=None if thermal else cfg['fixed_lambda'],**cfg['targets'][target]['params'])
    def states(x,t):
        return res.get_states(x,n_drop=0,bidir=False,current_temp=np.clip(t,*cfg['temperature_clip_K']) if thermal else None)
    xs = np.concatenate([states(x,t) for x,t,y in parts])
    model = MLPReadout(hidden_layer_sizes=(cfg['hidden_units'],), max_iter=cfg['epochs'],
                       alpha=cfg['weight_decay'],device_base_resistance=50.,random_state=seed)
    model.lr = cfg['learning_rate']
    model.fit(xs,limit-ys)
    pred = np.clip(limit-model.predict(states(xt,tt)),0,limit)
    folder.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(dict(True_RUL=yt,Pred_RUL=pred)).to_csv(folder/'predictions.csv',index=False)
    pd.DataFrame(dict(epoch=np.arange(1,len(model.loss_history)+1),loss=model.loss_history)).to_csv(folder/'loss.csv',index=False)
    (folder/'preprocessing.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    # Persist trained parameters and the reservoir input mapping.
    torch.save(model.model.state_dict(),folder/'MLP.pt')
    np.savez_compressed(folder/'reservoir.npz',**{f'input_{i}':v for i,v in enumerate(res._input_weights)})
    (folder/'model.json').write_text(json.dumps(dict(max_life=limit,target=target,arm=arm,seed=seed,config=cfg),indent=2),encoding='utf-8')
    return dict(target=target,arm=arm,seed=seed,**metrics(yt,pred))

def verify_data():
    """Validate packaged datasets against their SHA256 manifest."""
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    for entry in manifest:
        path=ROOT/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError(f'Data checksum mismatch: {path}')
    return len(manifest)

def summarize(rows, out):
    """Save per-run metrics and sample statistics across seeds."""
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'per_seed.csv',index=False)
    cols=['rmse_raw','rmse_normalized','nrmse_range','mae','r2']
    stats=frame.groupby(['target','arm'],sort=False)[cols].agg(['mean','std'])
    stats.columns=['_'.join(c) for c in stats.columns]
    stats.to_csv(out/'statistics.csv')

def run(cfg, out):
    """Execute the configuration and persist metrics, models, and run metadata."""
    verify_data()
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,cuda=torch.version.cuda,device='cuda' if torch.cuda.is_available() else 'cpu',threads=torch.get_num_threads())
    (out/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'src').glob('*.py'))+list((ROOT/'models').glob('*.py'))+[ROOT/'run.py',ROOT/'data/manifest.json']}
    (out/'code_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    rows=[]
    for target in cfg['targets']:
        for arm in cfg['arms']:
            for seed in cfg['seeds']:
                row=run_one(target,arm,seed,cfg,out/target/arm/f'seed{seed}')
                rows.append(row);summarize(rows,out)
                print(json.dumps(row),flush=True)
    verify_data()
    from .plotting import plot_predictions
    plot_predictions(out)
    (out/'COMPLETE.json').write_text(json.dumps(dict(runs=len(rows))),encoding='utf-8')
