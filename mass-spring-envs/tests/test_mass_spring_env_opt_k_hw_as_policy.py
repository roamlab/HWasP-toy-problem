import unittest
import gym
import numpy as np

import matplotlib.pyplot as plt

import mass_spring_envs
from mass_spring_envs.envs.mass_spring_env_opt_k_hw_as_policy import MassSpringEnv_OptK_HwAsPolicy

from shared_params import params

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
        self.env = MassSpringEnv_OptK_HwAsPolicy()
        obs = self.env.reset()

    
    def tearDown(self):
        # clean up after each test
        pass

    def test_rand_action(self):
        n_steps = 1000
        obs_arr = np.zeros(n_steps)
        for i in range(n_steps):
            if i % 100 ==0:
                action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            obs_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(obs_arr)

    def test_const_action(self):
        n_steps = 1000
        obs_arr = np.zeros(n_steps)
        for i in range(n_steps):
            action = [-(params.m1+params.m2)*params.g, 0]
            obs, reward, done, info = self.env.step(action)
            obs_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(obs_arr)


if __name__ == '__main__':
    unittest.main()