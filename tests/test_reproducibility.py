"""Verify dataset integrity, partitions, metrics, model sources, and input isolation."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiment import prepare,temperature_windows,verify_data
from src.metrics import metrics

class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8-sig'))
    def test_data_and_splits(self):
        self.assertEqual(verify_data(),7)
        for target,c in self.cfg['targets'].items():
            self.assertEqual(len(set(c['train'])),2)
            self.assertNotIn(c['test'],c['train'])
            self.assertTrue(all(p.split('_')[0]==c['test'].split('_')[0] for p in c['train']))
            for arm in self.cfg['arms']:
                parts,test,meta=prepare(target,arm,self.cfg)
                self.assertEqual(len(parts[0][2]),len(parts[1][2]))
                self.assertEqual(test[0].shape[2],1 if arm=='temperature_only' else 22)
                self.assertEqual(len(test[0]),len(test[2]))
                self.assertTrue(np.isfinite(test[0]).all())
    def test_metrics(self):
        m=metrics(np.array([0.,2.,4.]),np.array([1.,3.,5.]))
        self.assertEqual(m['rmse_raw'],1.)
        self.assertEqual(m['rmse_normalized'],.25)
        self.assertEqual(m['nrmse_range'],.25)
        self.assertAlmostEqual(m['r2'],.625)
    def test_model_hashes(self):
        expected={'MLP.py': '5da1de75e2c992dde53c1557a92cd64342c97a2f5cf555f2164742e2fc3b5221', 'reservoir.py': 'e99f1bf6d2fc53287593617f1a4a5ca19b4eab4337101cc7e99442840efd240d'}
        for name,digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest(),digest)

    def test_temperature_input_isolation(self):
        payload={'train': {'T_K': np.linspace(310.,350.,40)}}
        expected=temperature_windows(payload,330.,10.,self.cfg)
        payload['train']['X_vib']=np.full((40,21),np.nan)
        payload['train']['y_10s']=np.linspace(10000.,-10000.,40)
        actual=temperature_windows(payload,330.,10.,self.cfg)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(actual.shape,(21,20,1))

if __name__=='__main__':unittest.main()
