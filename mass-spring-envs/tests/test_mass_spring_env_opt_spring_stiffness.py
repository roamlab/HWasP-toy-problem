import unittest
import gym
import numpy as np

# import os
# print(os.environ['PYTHONPATH'])

import matplotlib.pyplot as plt

# import sys
# sys.path.append('.')
import mass_spring_envs
# from mass_spring_envs.envs.mass_spring_env_opt_spring_stiffness import MassSpringEnv_OptSpringStiffness

class TestMassSpringEnv_OptSpringStiffness(unittest.TestCase):
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
        self.env = gym.make("MassSpringEnv_OptSpringStiffness-v1")
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
            action = [-2.0*9.8, 0]
            obs, reward, done, info = self.env.step(action)
            obs_arr[i] = obs[0]
            if done:
                obs = self.env.reset()
        self.env.close()
        plt.figure()
        plt.plot(obs_arr)


if __name__ == '__main__':
    unittest.main()