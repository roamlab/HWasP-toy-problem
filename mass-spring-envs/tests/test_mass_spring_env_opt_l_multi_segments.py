import unittest
import numpy as np

import matplotlib.pyplot as plt

from mass_spring_envs.envs.mass_spring_env_opt_l_multi_segments import MassSpringEnv_OptL_MultiSegments_HwAsAction
from mass_spring_envs.envs.mass_spring_env_opt_l_multi_segments import MassSpringEnv_OptL_MultiSegments_HwAsPolicy

from shared_params import params_opt_l as params


class Test_MassSpringEnv_OptL_HwAsAction(unittest.TestCase):
    @classmethod
    def setupClass(cls):
        # runs once in class instantiation
        pass

    @classmethod
    def tearDownClass(cls):
        # runs once when class is torn down
        plt.show(block=True)


    def setUp(self):
        # everything in setup gets re instantiated for each test function
        self.env = MassSpringEnv_OptL_MultiSegments_HwAsAction(params)
        self.env.reset()

    
    def tearDown(self):
        # clean up after each test
        pass

    def test_rand_action(self):
        n_steps = 1000
        y1_arr = np.zeros(n_steps)
        for i in range(n_steps):
            if i % 200 ==0:
                action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            y1_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y1_arr)
        plt.title('hw_as_action:rand_actions')

    def test_const_action(self):
        n_steps = 1000
        y1_arr = np.zeros(n_steps)

        k = params.k
        y1 = 0.0
        l = 0.01
        m1 = params.m1
        m2 = params.m2
        g = params.g

        for i in range(n_steps):
            l_list = [l,] * params.n_segments
            action = [k*y1 - (m1+m2)*g] + l_list
            obs, reward, done, info = self.env.step(action)
            y1_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y1_arr)
        plt.title('hw_as_action:const_actions')

    def test_zero_action(self):
        n_steps = 1000
        y1_arr = np.zeros(n_steps)

        for i in range(n_steps):
            l_list = [0.0,] * params.n_segments
            action = [0.0] + l_list
            obs, reward, done, info = self.env.step(action)
            y1_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y1_arr)
        plt.title('hw_as_action:zero_actions')


class Test_MassSpringEnv_OptL_HwAsPolicy(unittest.TestCase):
    @classmethod
    def setupClass(cls):
        # runs once in class instantiation
        pass

    @classmethod
    def tearDownClass(cls):
        # runs once when class is torn down
        plt.show(block=True)


    def setUp(self):
        # everything in setup gets re instantiated for each test function
        # self.env = gym.make("MassSpringEnv_OptL_HwAsPolicy-v1")
        self.env = MassSpringEnv_OptL_MultiSegments_HwAsPolicy(params)
        obs = self.env.reset()

    
    def tearDown(self):
        # clean up after each test
        pass

    def test_rand_action(self):
        n_steps = 1000
        y1_arr = np.zeros(n_steps)
        for i in range(n_steps):
            if i % 100 ==0:
                action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            y1_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y1_arr)
        plt.title('hw_as_policy:rand_actions')

    def test_const_action(self):
        n_steps = 1000
        y1_arr = np.zeros(n_steps)

        k = params.k
        y1 = 0.0
        m1 = params.m1
        m2 = params.m2
        g = params.g

        for i in range(n_steps):
            action = [y1*k-m1*g, -m2*g + 0, 0]
            obs, reward, done, info = self.env.step(action)
            y1_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y1_arr)
        plt.title('hw_as_policy:const_actions')

    def test_zero_action(self):
        n_steps = 1000
        y1_arr = np.zeros(n_steps)

        for i in range(n_steps):
            action = [0, 0, 0]
            obs, reward, done, info = self.env.step(action)
            y1_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y1_arr)
        plt.title('hw_as_policy:zero_actions')

if __name__ == '__main__':
    unittest.main()