import unittest
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # only show warning and errors in TF

import tensorflow as tf

import matplotlib.pyplot as plt

from policies.opt_k_hw_as_policy.mech_policy_model import MechPolicyModel

from garage.tf.models import MLPModel
from policies.opt_k_hw_as_policy.comp_mech_policy import CompMechPolicy_OptK_HwAsPolicy
from garage.tf.envs import TfEnv
import gym
import mass_spring_envs

from shared_params import params
from mass_spring_envs.envs.mass_spring_env_opt_k_hw_as_action import MassSpringEnv_OptK_HwAsAction


class TestPolicy_OptK_HwAsPolicy(unittest.TestCase):
    @classmethod
    def setupClass(cls):
        # runs once in class instantiation
        pass

    @classmethod
    def tearDownClass(cls):
        # runs once when class is torn down
        pass


    def setUp(self):
        # everything in setup gets re-instantiated for each test function
        pass
    
    def tearDown(self):
        # clean up after each test
        tf.compat.v1.reset_default_graph()


    def test_mech_policy_model(self):
        y1 = 0.1
        v1 = 0.0
        f_normalized = 1.0

        mech_policy_model = MechPolicyModel(name='test_mech_policy_model', params=params)

        with tf.compat.v1.Session() as sess:
            f_ph = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
            y1_and_v1_ph = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.float32)
            pi_and_f_ts, log_std = mech_policy_model.build([f_ph, y1_and_v1_ph])
            output = sess.run(pi_and_f_ts, feed_dict={f_ph: [[f_normalized]], y1_and_v1_ph: [[y1, v1]]})
            # self.assertAlmostEqual(output[0][0], params.half_force_range*f_normalized-params.k_init*y1)
            self.assertAlmostEqual(output[0][1], params.half_force_range*f_normalized)


    def test_comp_mech_policy(self):
        y1 = 0.1
        v1 = 0.1
        
        env = TfEnv(MassSpringEnv_OptK_HwAsAction(params))

        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=(32, 32), 
            hidden_nonlinearity=tf.nn.tanh, 
            hidden_b_init=tf.zeros_initializer(), 
            hidden_w_init=tf.zeros_initializer(), 
            output_b_init=tf.zeros_initializer(), 
            output_w_init=tf.zeros_initializer())
        mech_policy_model = MechPolicyModel(params)

        with tf.compat.v1.Session() as sess:        
            comp_mech_policy = CompMechPolicy_OptK_HwAsPolicy(name='test_comp_mech_policy', 
                env_spec=env.spec, 
                comp_policy_model=comp_policy_model, 
                mech_policy_model=mech_policy_model)
                
            actions = comp_mech_policy.get_actions([[y1, v1]])
            print('actions: ', actions)
            # self.assertTrue(np.allclose(actions[1]['mean'], np.array([[params.half_force_range*0-params.k_init*y1, 0.0]])))
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[params.f_log_std_init, params.f_log_std_init]])))


            action = comp_mech_policy.get_action([y1, v1])
            print('single action: ', action)
            # self.assertTrue(np.allclose(action[1]['mean'], np.array([params.half_force_range*0-params.k_init*y1, 0.0])))
            self.assertTrue(np.allclose(action[1]['log_std'], np.array([params.f_log_std_init, params.f_log_std_init])))
            
            print(comp_mech_policy.distribution)


if __name__ == '__main__':
    unittest.main()