"""Validate reservoir state equations and temperature interfaces."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.reservoir import Reservoir

class ReservoirTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.RandomState(2).normal(size=(4,8,2))
    def reservoir(self,**kw):
        return Reservoir(n_internal_units=12,input_scaling=.3,**kw)
    def test_drop_matches_full_states(self):
        for temperature in [np.linspace(330,390,4), np.arange(32).reshape(4,8)+330.]:
            for noise in [0.,.0001]:
                np.random.seed(4);a=self.reservoir(noise_level=noise).get_states(self.x,current_temp=temperature,bidir=False)
                np.random.seed(4);b=self.reservoir(noise_level=noise).get_states(self.x,current_temp=temperature,bidir=False,n_drop=3)
                np.testing.assert_array_equal(b,a[:,3:])
                self.assertEqual(b.shape,(4,5,12))
    def test_drop_fixed_lambda(self):
        np.random.seed(4);a=self.reservoir(fixed_lambda=.1,Ea=0).get_states(self.x,bidir=False)
        np.random.seed(4);b=self.reservoir(fixed_lambda=.1,Ea=0).get_states(self.x,bidir=False,n_drop=7)
        np.testing.assert_array_equal(b,a[:,7:])
    def test_invalid_drop(self):
        for n in [-1,8,9,.5,True]:
            with self.assertRaises(ValueError):self.reservoir().get_states(self.x,n_drop=n)
    def test_constructor_temperature_matches_explicit(self):
        for t in [350.,np.linspace(330,390,4),np.arange(32).reshape(4,8)+330.]:
            for bidir in [False,True]:
                np.random.seed(2);a=self.reservoir(temperature=t,T_ref=375.15).get_states(self.x,n_drop=2,bidir=bidir)
                np.random.seed(2);b=self.reservoir(T_ref=375.15).get_states(self.x,current_temp=t,n_drop=2,bidir=bidir)
                np.testing.assert_array_equal(a,b)
    def test_temperature_changes_state(self):
        np.random.seed(2);a=self.reservoir(temperature=300.,Ea=0).get_states(self.x,bidir=False)
        np.random.seed(2);b=self.reservoir(temperature=400.,Ea=0).get_states(self.x,bidir=False)
        self.assertGreater(np.max(abs(a-b)),.01)
    def test_fixed_rate_no_implicit_thermal_bias(self):
        np.random.seed(2);a=self.reservoir(temperature=300.,fixed_lambda=.1).get_states(self.x,bidir=False)
        np.random.seed(2);b=self.reservoir(temperature=400.,fixed_lambda=.1).get_states(self.x,bidir=False)
        np.testing.assert_array_equal(a,b)
    def test_invalid_temperature(self):
        for t in [0.,-1.,float('nan'),np.ones(3),np.ones((4,7))]:
            with self.assertRaises(ValueError):self.reservoir().get_states(self.x,current_temp=t)
    def test_equation_and_bias_unchanged(self):
        t=np.linspace(330,390,4);np.random.seed(7)
        r=self.reservoir(T_ref=375.15,noise_level=0)
        actual=r.get_states(self.x,current_temp=t,bidir=False)
        rate=10/(.0048*np.exp(3242/t)+10)
        prev=np.zeros((4,12));expected=[]
        u=np.vstack(r._input_weights)
        for i in range(8):
            prev=(1-rate[:,None])*prev+rate[:,None]*(1+np.tanh(self.x[:,i]@u.T))
            expected.append(prev.copy())
        delta=.7/8.617e-5*(1/375.15-1/t)
        expected=np.stack(expected,axis=1)+np.log(np.exp(delta)+1e-9)[:,None,None]
        np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-14)

if __name__=='__main__':unittest.main()
