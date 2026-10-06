"""Run repository-configured bearing RUL experiments."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from src.experiment import ROOT, run, verify_data

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=ROOT/'configs/experiment.json')
    p.add_argument('--targets',nargs='+')
    p.add_argument('--arms',nargs='+',choices=['multimodal','vibration_only','temperature_only'])
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--smoke',action='store_true',help='2 epochs; pipeline test only, not scientific results')
    p.add_argument('--verify',action='store_true',help='Check bundled dataset hashes and exit')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.verify:
        print(f'Verified {verify_data()} datasets');return
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    if a.targets:cfg['targets']={t:cfg['targets'][t] for t in a.targets}
    if a.arms:cfg['arms']=a.arms
    if a.seeds is not None:cfg['seeds']=a.seeds
    if a.smoke:cfg.update(epochs=2,seeds=[0],smoke_test=True)
    out=a.output or ROOT/'results'/('smoke' if a.smoke else 'runs')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run(cfg,out.resolve())

if __name__=='__main__':main()
