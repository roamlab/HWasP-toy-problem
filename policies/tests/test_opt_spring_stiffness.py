import unittest
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # only show warning and errors in TF

import tensorflow as tf

import matplotlib.pyplot as plt

from policies.opt_spring_stiffness.mech_policy_model import MechPolicyModel

from garage.tf.models import MLPModel
from policies.opt_spring_stiffness.policy import CompMechPolicy_OptSpringStiffness
from garage.tf.envs import TfEnv
import gym
import mass_spring_envs


def inv_sigmoid(y, lb, ub):
    '''
    The inverse of the sigmoid scaling transform y = 1/(1+exp(-x)) * (ub - lb) + lb
    '''
    return -np.log((ub - lb)/ (y - lb) - 1)



class TestPolicy_OptSpringStiffness(unittest.TestCase):
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
        k = 5e1
        ub = 1e2
        lb = 0.0
        k_pre_init = inv_sigmoid(k, lb, ub)

        y1 = 0.1
        f = 10.0

        log_std_init = 0.1

        mech_policy_model = MechPolicyModel(name='test_mech_policy_model', k_pre_init=k_pre_init, log_std_init=log_std_init)

        with tf.compat.v1.Session() as sess:
            f_ph = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
            y1_ph = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
            pi_and_f_ts, log_std = mech_policy_model.build([f_ph, y1_ph])
            output = sess.run(pi_and_f_ts, feed_dict={f_ph: [[f]], y1_ph: [[y1]]})
            self.assertAlmostEqual(output[0][0], f-k*y1)
            self.assertAlmostEqual(output[0][1], f)


    def test_comp_mech_policy(self):
        k = 5e1
        ub = 1e2
        lb = 0.0
        k_pre = inv_sigmoid(k, lb, ub)

        y1 = 0.1
        log_std_init = 0.1
        
        env = TfEnv(gym.make('MassSpringEnv_OptSpringStiffness-v1'))

        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=(32, 32), 
            hidden_nonlinearity=tf.nn.tanh, 
            hidden_b_init=tf.zeros_initializer(), 
            hidden_w_init=tf.zeros_initializer(), 
            output_b_init=tf.zeros_initializer(), 
            output_w_init=tf.zeros_initializer())
        mech_policy_model = MechPolicyModel(k_pre_init=k_pre, log_std_init=log_std_init)

        LARGE_NUMBER = 1e6

        with tf.compat.v1.Session() as sess:        
            comp_mech_policy = CompMechPolicy_OptSpringStiffness(name='test_comp_mech_policy', 
                env_spec=env.spec, 
                comp_policy_model=comp_policy_model, 
                mech_policy_model=mech_policy_model)
            actions = comp_mech_policy.get_actions([[y1]])
            print('actions: ', actions)
            self.assertAlmostEqual(actions[0][0][1], 0.0)
            self.assertTrue(np.allclose(actions[1]['mean'], np.array([[0-k*y1, 0.0]])))
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[0.1, -LARGE_NUMBER]])))


            action = comp_mech_policy.get_action([y1])
            print('single action: ', action)
            self.assertAlmostEqual(action[0][1], 0.0)
            self.assertTrue(np.allclose(action[1]['mean'], np.array([0-k*y1, 0.0])))
            self.assertTrue(np.allclose(action[1]['log_std'], np.array([0.1, -LARGE_NUMBER])))
            
            print(comp_mech_policy.distribution)


if __name__ == '__main__':
    unittest.main()