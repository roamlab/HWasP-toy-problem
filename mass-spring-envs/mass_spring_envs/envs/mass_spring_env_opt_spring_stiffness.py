'''
illustration of the system:

        --------------
        ^      \
        |       \
        |       /  spring, stiffness k, masless, zero natural length 
        |      /
        |      \
        |       \
        |    +-----+
        |    |     |
        |    |     |  point mass m1, position y1
 dist h |    +--+--+
        |       |
        |       | bar, length l, massless
        |       |
        |    +--+--+
        |    |     |
        |    |     |  point mass m2, position y2
        |    +-----+
        |                               
        |               |               |
        |               |               |
        |               |               |
        v      Goal     v input f       v g
        --------------

'''

import gym
import numpy as np

# in SI
ACT_RANGE = 5e1 # force in N
OBS_RANGE = 5e0 # pos in m
REWARD_RANGE = 1e2
m1 = 1.0
m2 = 1.0
h = 2.0
l = 1.0
g = 9.8
dt = 0.001


class MassSpringEnv_OptSpringStiffness(gym.Env):
    '''
    1D mass-spring toy problem.
    Case I: optimizing the spring stiffness k
    Overall policy: pi = f - k*y1
    Action: pi, F(pass-in F for reward calculation)
    observation: y1 (or y2)
    '''

    def __init__(self):
        self.action_space = gym.spaces.Box(low=-ACT_RANGE, high=ACT_RANGE, shape=(2, ), dtype=np.float64) # 1st: redifined action pi, 2nd: original action F
        self.observation_space = gym.spaces.Box(low=0.0, high=OBS_RANGE, shape=(1, ), dtype=np.float64) # obs y1

        self.v = 0.0 # vel of both masses
        self.y1 = 0.0

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        pi = action[0] # overall policy output
        f_total = pi + (m1+ m2) * g
        a = f_total / (m1+ m2)
        self.v = self.v + a * dt # simple Euler integration
        self.y1 = self.y1 + self.v * dt

        y2 = self.y1 + l

        obs = np.array([self.y1])

        f = action[1] # additional info for reward calculation
        alpha = 10.0
        beta = 1.0
        reward = -alpha * (y2 - h)**2 - beta*f**2

        if self.y1 < 0.0 or y2 > OBS_RANGE:
            done = True
        else:
            done = False
        
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.v = 0.0
        self.y1 = 0.0
        return np.array([self.y1])
    

    def render(self, mode='human'):
        pass
