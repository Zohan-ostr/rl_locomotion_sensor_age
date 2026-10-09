"""Causality, identity, physical constraints and joint derivative consistency."""
import json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from linear_compensator import FeatureHistory,LinearCompensator
from sensor_age import Packet


def packets(sample=0,delivery=0):
    return {'joints':Packet('joints',sample,delivery,{'q':np.zeros(10),'dq':np.ones(10)*.2}),
            'imu':Packet('imu',sample,delivery,{'omega':np.array([0.,0.,1.]),'g':np.array([1.,0.,0.])})}


class M3Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'model.npz'
        h=FeatureHistory(False);h.update(packets(),0);h.context(10,np.zeros(10),np.zeros(3),0)
        model={'metadata':json.dumps({'actions':False})}
        for stream in ('joints','imu'):
            n=len(h.features(stream,10,.02));model[stream+'_mean']=np.zeros(n);model[stream+'_std']=np.ones(n)
            model[stream+'_weights']=np.zeros((2*(n+1) if stream=='joints' else n+1,10 if stream=='joints' else 6))
        np.savez_compressed(self.path,**model)
        self.adapter=LinearCompensator(np.tile([-2.,2.],(10,1)),{'joints':.08,'imu':.08},{'model_path':str(self.path),'gain':1})
    def tearDown(self):self.temp.cleanup()
    def test_zero_age_exact_even_outside_limits(self):
        p=packets();p['joints']=Packet('joints',0,0,{'q':np.ones(10)*3,'dq':np.ones(10)*10})
        self.adapter.update(p,0);self.adapter.context(0,np.zeros(10),np.zeros(3),0)
        out=self.adapter.predict(0)
        for s in p:
            for k,v in p[s].payload.items():np.testing.assert_array_equal(out[s][k],v)
    def test_future_and_old_packets(self):
        self.adapter.update(packets(10,10),10)
        self.adapter.update(packets(),10)
        self.assertEqual(self.adapter.history['joints'][-1].sample_step,10)
        with self.assertRaises(ValueError):self.adapter.update(packets(20,30),10)
        self.assertEqual(self.adapter.history['joints'][-1].sample_step,10)
    def test_growth_and_horizon_cap_constant_velocity(self):
        self.adapter.update(packets(),0)
        out=[]
        for step in (10,40,60,100):
            self.adapter.context(step,np.zeros(10),np.zeros(3),0);out.append(self.adapter.predict(step))
        np.testing.assert_allclose(out[0]['joints']['q'],.004)
        np.testing.assert_allclose(out[1]['joints']['q'],.016)
        for i in (2,3):np.testing.assert_array_equal(out[i]['joints']['q'],out[1]['joints']['q'])
    def test_joint_curve_analytic_derivative_including_coefficient_bounds(self):
        for coefficients in (np.ones((2,10))*.001,np.ones((2,10))*10):
            packet=packets()['joints'];t=.04;eps=1e-6
            numerical=(self.adapter.joint_curve(packet,coefficients,t+eps)['q']-self.adapter.joint_curve(packet,coefficients,t-eps)['q'])/(2*eps)
            np.testing.assert_allclose(numerical,self.adapter.joint_curve(packet,coefficients,t)['dq'],rtol=1e-8,atol=1e-8)
    def test_clipped_joint_derivative_and_gravity(self):
        p=packets();p['joints']=Packet('joints',0,0,{'q':np.ones(10)*1.99,'dq':np.ones(10)})
        self.adapter.update(p,0);self.adapter.context(40,np.zeros(10),np.zeros(3),0);out=self.adapter.predict(40)
        np.testing.assert_array_equal(out['joints']['q'],np.ones(10)*2)
        np.testing.assert_array_equal(out['joints']['dq'],np.zeros(10))
        self.assertAlmostEqual(np.linalg.norm(out['imu']['g']),1)
        self.assertLess(out['imu']['g'][1],0)
    def test_no_actions_variant_invariance(self):
        a=FeatureHistory(False);b=FeatureHistory(False)
        for h in (a,b):h.update(packets(),0)
        a.context(10,np.zeros(10),np.ones(3),.2);b.context(10,np.ones(10)*100,np.ones(3),.2)
        np.testing.assert_array_equal(a.features('joints',10,.02),b.features('joints',10,.02))
    def test_context_and_timestamp_features(self):
        h=FeatureHistory(True);h.update(packets(),0);h.context(10,np.ones(10),np.zeros(3),0)
        self.assertEqual(h.action_history[-1][0],0)
        np.testing.assert_array_equal(h.features('imu',10,.02)[-10:],np.zeros(10))
        with self.assertRaises(ValueError):h.context(0,np.zeros(10),np.zeros(3),0)
    def test_bounded_history_and_payload_unchanged(self):
        for step in range(0,100,10):self.adapter.update(packets(step,step),step)
        self.assertEqual(len(self.adapter.history['joints']),3)
        p=self.adapter.history['joints'][-1];self.adapter.context(100,np.zeros(10),np.zeros(3),0)
        self.adapter.predict(100);np.testing.assert_array_equal(p.payload['q'],np.zeros(10))

if __name__=='__main__':unittest.main(verbosity=2)
