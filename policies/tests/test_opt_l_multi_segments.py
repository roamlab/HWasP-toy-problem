import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # only show warning and errors in TF
import tensorflow as tf
import matplotlib.pyplot as plt
from shared_params import params_opt_l as params

from garage.tf.envs import TfEnv
from garage.tf.models import MLPModel

from mass_spring_envs.envs.mass_spring_env_opt_l_multi_segments import MassSpringEnv_OptL_MultiSegments_HwAsAction
from mass_spring_envs.envs.mass_spring_env_opt_l_multi_segments import MassSpringEnv_OptL_MultiSegments_HwAsPolicy

from policies.opt_l_multi_segments.models import MechPolicyModel_OptL_MultiSegments_HwAsAction
from policies.opt_l_multi_segments.models import MechPolicyModel_OptL_MultiSegments_HwAsPolicy

from policies.opt_l_multi_segments.policies import CompMechPolicy_OptL_MultiSegments_HwAsAction
from policies.opt_l_multi_segments.policies import CompMechPolicy_OptL_MultiSegments_HwAsPolicy      


class TestPolicy_OptL_MultiSegments_HwAsAction(unittest.TestCase):
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
        print('\n Testing MechPolicyModel_OptL_MultiSegments_HwAsAction ...')
        y1 = 0.1
        v1 = 0.0

        mech_policy_model = MechPolicyModel_OptL_MultiSegments_HwAsAction(name='test_mech_policy_model', params=params)

        with tf.compat.v1.Session() as sess:
            y1_and_v1_ph = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.float32)
            l_ts, log_std = mech_policy_model.build(y1_and_v1_ph)
            output = sess.run([l_ts, log_std], feed_dict={y1_and_v1_ph: [[y1, v1]]})
            print(output)
            self.assertTrue(np.allclose(output[1][0], np.array([params.f_log_std_init_action] + [params.l_log_std_init_action,] * params.n_segments), atol=1e-3))


    def test_comp_mech_policy(self):
        print('\n Testing CompMechPolicy_OptL_MultiSegments_HwAsAction ...')
        y1 = 0.1
        v1 = 0.1

        env = TfEnv(MassSpringEnv_OptL_MultiSegments_HwAsAction(params))

        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=(32, 32), 
            hidden_nonlinearity=tf.nn.tanh, 
            hidden_b_init=tf.zeros_initializer(), 
            hidden_w_init=tf.zeros_initializer(), 
            output_b_init=tf.zeros_initializer(), 
            output_w_init=tf.zeros_initializer())
        mech_policy_model = MechPolicyModel_OptL_MultiSegments_HwAsAction(params=params)

        with tf.compat.v1.Session() as sess:        
            comp_mech_policy = CompMechPolicy_OptL_MultiSegments_HwAsAction(name='test_comp_mech_policy', 
                env_spec=env.spec, 
                comp_policy_model=comp_policy_model, 
                mech_policy_model=mech_policy_model)

            actions = comp_mech_policy.get_actions([[y1, v1]])
            print('actions: ', actions)
            # self.assertTrue(np.allclose(actions[1]['mean'], np.array([[0.0,] + [params.l_init,] * params.n_segments]))) # uncomment if l is initialized with l_init, not randomly
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[params.f_log_std_init_action] + [params.l_log_std_init_action,] * params.n_segments]), atol=1e-3))

            action = comp_mech_policy.get_action([y1, v1])
            print('single action: ', action)
            # self.assertTrue(np.allclose(actions[1]['mean'], np.array([0.0,] + [params.l_init,] * params.n_segments))) # uncomment if l is initialized with l_init, not randomly
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[params.f_log_std_init_action] + [params.l_log_std_init_action,] * params.n_segments]), atol=1e-3))
            
            print(comp_mech_policy.distribution)


class TestPolicy_OptL_MultiSegments_HwAsPolicy(unittest.TestCase):
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
        print('\n Testing MechPolicyModel_OptL_MultiSegments_HwAsPolicy ...')
        y1 = 0.1
        v1 = 0.0

        y2 = 0.2
        v2 = 0.0

        f_normalized = 1.0

        mech_policy_model = MechPolicyModel_OptL_MultiSegments_HwAsPolicy(name='test_mech_policy_model', params=params)

        with tf.compat.v1.Session() as sess:
            f_ph = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
            y1_v1_y2_v2_ph = tf.compat.v1.placeholder(shape=(None, 4), dtype=tf.float32)
            f1_f2_f_ts, log_std = mech_policy_model.build([f_ph, y1_v1_y2_v2_ph])
            output = sess.run([f1_f2_f_ts, log_std], feed_dict={f_ph: [[f_normalized]], y1_v1_y2_v2_ph: [[y1, v1, y2, v2]]})

            print(output)

            expected_yl = 1/2 * (y1 + y2)
            expected_vl = 1/2 * (v1 + v2)
            expected_f1 = params.k_interface * (expected_yl - params.l_init * params.n_segments / 2 - y1) + params.b_interface * (expected_vl - v1)
            expected_f2 = - expected_f1

            # self.assertAlmostEqual(output[0][0][0], expected_f1, places=3) # uncomment if l is initialized with l_init, not randomly
            # self.assertAlmostEqual(output[0][0][1], expected_f2, places=3) # uncomment if l is initialized with l_init, not randomly
            # self.assertAlmostEqual(output[0][0][2], f_normalized*params.half_force_range, places=3) # uncomment if l is initialized with l_init, not randomly

            self.assertAlmostEqual(output[0][0][2], params.half_force_range*f_normalized, places=3)
            self.assertAlmostEqual(output[1][0][2], params.f_log_std_init_auxiliary, places=3)


    def test_comp_mech_policy(self):        
        y1 = 0.1
        v1 = 0.1
        
        y2 = 0.2
        v2 = 0.1

        expected_yl = 1/2 * (y1 + y2)
        expected_vl = 1/2 * (v1 + v2)
        expected_f1 = params.k_interface * (expected_yl - params.l_init * params.n_segments / 2 - y1) + params.b_interface * (expected_vl - v1)
        expected_f2 = - expected_f1

        env = TfEnv(MassSpringEnv_OptL_MultiSegments_HwAsPolicy(params))

        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=(32, 32), 
            hidden_nonlinearity=tf.nn.tanh, 
            hidden_b_init=tf.zeros_initializer(), 
            hidden_w_init=tf.zeros_initializer(), 
            output_b_init=tf.zeros_initializer(), 
            output_w_init=tf.zeros_initializer())
        mech_policy_model = MechPolicyModel_OptL_MultiSegments_HwAsPolicy(params)


        with tf.compat.v1.Session() as sess:        
            comp_mech_policy = CompMechPolicy_OptL_MultiSegments_HwAsPolicy(name='test_comp_mech_policy', 
                env_spec=env.spec, 
                comp_policy_model=comp_policy_model, 
                mech_policy_model=mech_policy_model)
                
            actions = comp_mech_policy.get_actions([[y1, v1, y2, v2]])
            print('actions: ', actions)
            # self.assertTrue(np.allclose(actions[1]['mean'], np.array([[expected_f1, expected_f2, 0.0]]))) # uncomment if l is initialized with l_init, not randomly
            self.assertTrue(np.allclose(actions[1]['log_std'], np.array([[params.f_log_std_init_action, params.f_log_std_init_action, params.f_log_std_init_auxiliary]])))


            action = comp_mech_policy.get_action([y1, v1, y2, v2])
            print('single action: ', action)
            # self.assertTrue(np.allclose(action[1]['mean'], np.array([expected_f1, expected_f2, 0.0]))) # uncomment if l is initialized with l_init, not randomly
            self.assertTrue(np.allclose(action[1]['log_std'], np.array([params.f_log_std_init_action, params.f_log_std_init_action, params.f_log_std_init_auxiliary])))
            
            print(comp_mech_policy.distribution)



if __name__ == '__main__':
    unittest.main()