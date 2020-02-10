import unittest
import numpy as np

import matplotlib.pyplot as plt

from mass_spring_envs.envs.mass_spring_env_opt_k import MassSpringEnv_OptK_HwAsAction
from mass_spring_envs.envs.mass_spring_env_opt_k import MassSpringEnv_OptK_HwAsPolicy

from shared_params import params


class Test_MassSpringEnv_OptK_HwAsAction(unittest.TestCase):
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
        self.env = MassSpringEnv_OptK_HwAsAction(params)
        self.env.reset()

    
    def tearDown(self):
        # clean up after each test
        pass

    def test_rand_action(self):
        n_steps = 1000
        y2_arr = np.zeros(n_steps)
        for i in range(n_steps):
            if i % 200 ==0:
                action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            y2_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y2_arr)
        plt.title('hw_as_action:rand_actions')

    def test_const_action(self):
        n_steps = 1000
        y2_arr = np.zeros(n_steps)

        y1 = 0.1
        k = 50.0
        m1 = params.m1
        m2 = params.m2
        g = params.g

        for i in range(n_steps):
            action = [k*y1 - (m1+m2)*g, k]
            obs, reward, done, info = self.env.step(action)
            y2_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y2_arr)
        plt.title('hw_as_action:const_actions')

    def test_zero_action(self):
        n_steps = 1000
        y2_arr = np.zeros(n_steps)

        for i in range(n_steps):
            action = [0, 0]
            obs, reward, done, info = self.env.step(action)
            y2_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y2_arr)
        plt.title('hw_as_action:zero_actions')


class Test_MassSpringEnv_OptK_HwAsPolicy(unittest.TestCase):
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
        # self.env = gym.make("MassSpringEnv_OptK_HwAsPolicy-v1")
        self.env = MassSpringEnv_OptK_HwAsPolicy(params)
        obs = self.env.reset()

    
    def tearDown(self):
        # clean up after each test
        pass

    def test_rand_action(self):
        n_steps = 1000
        y2_arr = np.zeros(n_steps)
        for i in range(n_steps):
            if i % 100 ==0:
                action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            y2_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y2_arr)
        plt.title('hw_as_policy:rand_actions')

    def test_const_action(self):
        n_steps = 1000
        y2_arr = np.zeros(n_steps)
        for i in range(n_steps):
            action = [-(params.m1+params.m2)*params.g, 0]
            obs, reward, done, info = self.env.step(action)
            y2_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y2_arr)
        plt.title('hw_as_policy:const_actions')

    def test_zero_action(self):
        n_steps = 1000
        y2_arr = np.zeros(n_steps)

        for i in range(n_steps):
            action = [0, 0]
            obs, reward, done, info = self.env.step(action)
            y2_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(y2_arr)
        plt.title('hw_as_policy:zero_actions')

if __name__ == '__main__':
    unittest.main()