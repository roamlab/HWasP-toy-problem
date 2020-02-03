import unittest
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # only show warning and errors in TF

import tensorflow as tf

import matplotlib.pyplot as plt

from policies.opt_k_hw_as_action.mech_policy_model import MechPolicyModel

from garage.tf.models import MLPModel
from policies.opt_k_hw_as_action.comp_mech_policy import CompMechPolicy_OptK_HwAsAction
from garage.tf.envs import TfEnv
import gym
import mass_spring_envs

from shared_params import params
from shared_params.params import inv_sigmoid

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

        mech_policy_model = MechPolicyModel(name='test_mech_policy_model', k_pre_init=params.k_pre_init, log_std_init=params.log_std_init)

        with tf.compat.v1.Session() as sess:
            y1_and_v1_ph = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.float32)
            k_ts, log_std = mech_policy_model.build(y1_and_v1_ph)
            output = sess.run([k_ts, log_std], feed_dict={y1_and_v1_ph: [[y1, v1]]})
            # print(output)
            self.assertAlmostEqual(output[0][0][0], params.k_init)
            self.assertAlmostEqual(output[1][0][0], params.log_std_init)



    def test_comp_mech_policy(self):
        y1 = 0.1
        v1 = 0.1

        env = TfEnv(gym.make('MassSpringEnv_OptK_HwAsPolicy-v1'))

        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=(32, 32), 
            hidden_nonlinearity=tf.nn.tanh, 
            hidden_b_init=tf.zeros_initializer(), 
            hidden_w_init=tf.zeros_initializer(), 
            output_b_init=tf.zeros_initializer(), 
            output_w_init=tf.zeros_initializer())
        mech_policy_model = MechPolicyModel(k_pre_init=params.k_pre_init, log_std_init=params.log_std_init)

        with tf.compat.v1.Session() as sess:        
            comp_mech_policy = CompMechPolicy_OptK_HwAsAction(name='test_comp_mech_policy', 
                env_spec=env.spec, 
                comp_policy_model=comp_policy_model, 
                mech_policy_model=mech_policy_model)

            actions = comp_mech_policy.get_actions([[y1, v1]])
            print('actions: ', actions)
            self.assertTrue(np.allclose(actions[1]['mean'], np.array([[0.0, params.k_init]])))
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[params.log_std_init, params.log_std_init]])))

            action = comp_mech_policy.get_action([y1, v1])
            print('single action: ', action)
            self.assertTrue(np.allclose(actions[1]['mean'], np.array([0.0, params.k_init])))
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([params.log_std_init, params.log_std_init])))
            
            print(comp_mech_policy.distribution)



if __name__ == '__main__':
    unittest.main()