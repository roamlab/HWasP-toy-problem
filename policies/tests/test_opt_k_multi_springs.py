import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # only show warning and errors in TF
import tensorflow as tf
import matplotlib.pyplot as plt
from shared_params import params

from garage.tf.envs import TfEnv
from garage.tf.models import MLPModel

from mass_spring_envs.envs.mass_spring_env_opt_k_multi_springs import MassSpringEnv_OptK_MultiSprings_HwAsAction
from mass_spring_envs.envs.mass_spring_env_opt_k_multi_springs import MassSpringEnv_OptK_MultiSprings_HwAsPolicy

from policies.opt_k_multi_springs.models import MechPolicyModel_OptK_MultiSprings_HwAsAction
from policies.opt_k_multi_springs.models import MechPolicyModel_OptK_MultiSprings_HwAsPolicy
from policies.opt_k_multi_springs.models import CompMechPolicyModel_OptK_MultiSprings_HwInPolicyAndAction

from policies.opt_k_multi_springs.policies import CompMechPolicy_OptK_MultiSprings_HwAsAction
from policies.opt_k_multi_springs.policies import CompMechPolicy_OptK_MultiSprings_HwAsPolicy      
from policies.opt_k_multi_springs.policies import CompMechPolicy_OptK_MultiSprings_HwInPolicyAndAction     

class TestPolicy_OptK_MultiSprings_HwAsAction(unittest.TestCase):
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
        print('\n Testing MechPolicyModel_OptK_MultiSprings_HwAsAction ...')
        y1 = 0.1
        v1 = 0.0

        mech_policy_model = MechPolicyModel_OptK_MultiSprings_HwAsAction(name='test_mech_policy_model', params=params)

        with tf.compat.v1.Session() as sess:
            y1_and_v1_ph = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.float32)
            k_ts, log_std = mech_policy_model.build(y1_and_v1_ph)
            output = sess.run([k_ts, log_std], feed_dict={y1_and_v1_ph: [[y1, v1]]})
            print(output)
            self.assertTrue(np.allclose(output[1][0], np.array([params.f_log_std_init_action] + [params.k_log_std_init_action,] * params.n_springs), atol=1e-3))


    def test_comp_mech_policy(self):
        print('\n Testing CompMechPolicy_OptK_MultiSprings_HwAsAction ...')
        y1 = 0.1
        v1 = 0.1

        env = TfEnv(MassSpringEnv_OptK_MultiSprings_HwAsAction(params))

        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=(32, 32), 
            hidden_nonlinearity=tf.nn.tanh, 
            hidden_b_init=tf.zeros_initializer(), 
            hidden_w_init=tf.zeros_initializer(), 
            output_b_init=tf.zeros_initializer(), 
            output_w_init=tf.zeros_initializer())
        mech_policy_model = MechPolicyModel_OptK_MultiSprings_HwAsAction(params=params)

        with tf.compat.v1.Session() as sess:        
            comp_mech_policy = CompMechPolicy_OptK_MultiSprings_HwAsAction(name='test_comp_mech_policy', 
                env_spec=env.spec, 
                comp_policy_model=comp_policy_model, 
                mech_policy_model=mech_policy_model)

            actions = comp_mech_policy.get_actions([[y1, v1]])
            print('actions: ', actions)
            # self.assertTrue(np.allclose(actions[1]['mean'], np.array([[0.0, params.k_init]])))
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[params.f_log_std_init_action] + [params.k_log_std_init_action,] * params.n_springs]), atol=1e-3))

            action = comp_mech_policy.get_action([y1, v1])
            print('single action: ', action)
            # self.assertTrue(np.allclose(actions[1]['mean'], np.array([0.0, params.k_init])))
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[params.f_log_std_init_action] + [params.k_log_std_init_action,] * params.n_springs]), atol=1e-3))
            
            print(comp_mech_policy.distribution)


class TestPolicy_OptK_MultiSprings_HwAsPolicy(unittest.TestCase):
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
        print('\n Testing MechPolicyModel_OptK_MultiSprings_HwAsPolicy ...')
        y1 = 0.1
        v1 = 0.0
        f_normalized = 1.0

        mech_policy_model = MechPolicyModel_OptK_MultiSprings_HwAsPolicy(name='test_mech_policy_model', params=params)

        with tf.compat.v1.Session() as sess:
            f_ph = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
            y1_and_v1_ph = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.float32)
            pi_and_f_ts, log_std = mech_policy_model.build([f_ph, y1_and_v1_ph])
            output = sess.run([pi_and_f_ts, log_std], feed_dict={f_ph: [[f_normalized]], y1_and_v1_ph: [[y1, v1]]})
            print(output)
            # self.assertAlmostEqual(output[0][0][0], params.half_force_range*f_normalized-params.n_springs * params.k_init*y1)
            self.assertAlmostEqual(output[0][0][1], params.half_force_range*f_normalized)
            self.assertAlmostEqual(output[1][0][1], params.f_log_std_init_auxiliary)


    def test_comp_mech_policy(self):        
        y1 = 0.1
        v1 = 0.1
        
        env = TfEnv(MassSpringEnv_OptK_MultiSprings_HwAsPolicy(params))

        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=(32, 32), 
            hidden_nonlinearity=tf.nn.tanh, 
            hidden_b_init=tf.zeros_initializer(), 
            hidden_w_init=tf.zeros_initializer(), 
            output_b_init=tf.zeros_initializer(), 
            output_w_init=tf.zeros_initializer())
        mech_policy_model = MechPolicyModel_OptK_MultiSprings_HwAsPolicy(params)

        with tf.compat.v1.Session() as sess:        
            comp_mech_policy = CompMechPolicy_OptK_MultiSprings_HwAsPolicy(name='test_comp_mech_policy', 
                env_spec=env.spec, 
                comp_policy_model=comp_policy_model, 
                mech_policy_model=mech_policy_model)
                
            actions = comp_mech_policy.get_actions([[y1, v1]])
            print('actions: ', actions)
            # self.assertTrue(np.allclose(actions[1]['mean'], np.array([[params.half_force_range*0-params.k_init*y1, 0.0]])))
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[params.f_log_std_init_action, params.f_log_std_init_auxiliary]])))


            action = comp_mech_policy.get_action([y1, v1])
            print('single action: ', action)
            # self.assertTrue(np.allclose(action[1]['mean'], np.array([params.half_force_range*0-params.k_init*y1, 0.0])))
            self.assertTrue(np.allclose(action[1]['log_std'], np.array([params.f_log_std_init_action, params.f_log_std_init_auxiliary])))
            
            print(comp_mech_policy.distribution)


class TestPolicy_OptK_MultiSprings_HwInPolicyAndAction(unittest.TestCase):
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


    def test_comp_mech_policy_model(self):
        print('\n Testing CompMechPolicyModel_OptK_MultiSprings_HwInPolicyAndAction ...')
        y1 = 0.1
        v1 = 0.0

        comp_mech_policy_model = CompMechPolicyModel_OptK_MultiSprings_HwInPolicyAndAction(name='test_comp_mech_policy_model', params=params)

        with tf.compat.v1.Session() as sess:
            y1_and_v1_ph = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.float32)
            f_and_k_ts, log_std = comp_mech_policy_model.build(y1_and_v1_ph)
            output = sess.run([f_and_k_ts, log_std], feed_dict={y1_and_v1_ph: [[y1, v1]]})
            print(output)
            self.assertAlmostEqual(output[1][0][0], params.f_log_std_init_action, places=3)
            self.assertAlmostEqual(output[1][0][1], params.k_log_std_init_auxiliary, places=3)


    def test_comp_mech_policy(self):
        print('\n Testing CompMechPolicy_OptK_MultiSprings_HwInPolicyAndAction ...')        
        y1 = 0.1
        v1 = 0.1

        env = TfEnv(MassSpringEnv_OptK_MultiSprings_HwAsAction(params))

        comp_mech_policy_model = CompMechPolicyModel_OptK_MultiSprings_HwInPolicyAndAction(params)

        with tf.compat.v1.Session() as sess:        
            comp_mech_policy = CompMechPolicy_OptK_MultiSprings_HwInPolicyAndAction(name='test_comp_mech_policy', 
                env_spec=env.spec, 
                comp_mech_policy_model=comp_mech_policy_model)

            actions = comp_mech_policy.get_actions([[y1, v1]])
            print('actions: ', actions)
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[params.f_log_std_init_action,] + [params.k_log_std_init_auxiliary,] * params.n_springs])))

            action = comp_mech_policy.get_action([y1, v1])
            print('single action: ', action)
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([params.f_log_std_init_action,] + [params.k_log_std_init_auxiliary,] * params.n_springs)))
            
            print(comp_mech_policy.distribution)


if __name__ == '__main__':
    unittest.main()