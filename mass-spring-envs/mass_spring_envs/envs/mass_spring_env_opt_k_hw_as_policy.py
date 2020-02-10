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

from shared_params import params

class MassSpringEnv_OptK_HwAsPolicy(gym.Env):
    '''
    1D mass-spring toy problem.
    Case I: optimizing the spring stiffness k
    Overall policy: pi = f - k*y1
    Action: pi, f(pass-in f for reward calculation)
    observation: y1, v1
    '''

    def __init__(self, params):

        self.half_force_range = params.half_force_range
        self.k_lb = params.k_lb
        self.k_ub = params.k_ub
        self.pos_range = params.pos_range
        self.half_vel_range = params.half_vel_range
        self.m1 = params.m1
        self.m2 = params.m2
        self.h = params.h
        self.l = params.l
        self.g = params.g
        self.dt = params.dt
        self.n_steps_per_action = params.n_steps_per_action
        self.reward_alpha = params.reward_alpha
        self.reward_beta = params.reward_beta
        self.reward_gamma = params.reward_gamma


        self.action_space = gym.spaces.Box(low=-self.half_force_range, high=self.half_force_range, shape=(2, ), dtype=np.float32) # 1st: redifined action pi, 2nd: original action f
        self.observation_space = gym.spaces.Box(low=np.array([0, -self.half_vel_range]), high=np.array([self.pos_range, self.half_vel_range]), dtype=np.float32) # obs y1 and v1
        
        # states
        # self.v1 = 0.0 # vel of both masses
        # self.y1 = 0.0
        self.v1 = np.random.uniform(-self.half_vel_range, self.half_vel_range) # vel of both masses
        self.y1 = np.random.uniform(0, self.pos_range)
        self.step_cnt = 0

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the policy, here a combination of redefined policy action (pi) and additional info (f) for reward calculation
        
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """

        action = np.clip(action.copy(), self.action_space.low, self.action_space.high)

        pi = action[0] # redifined policy output
        f = action[1]  # additional info for reward calculation



        f_total = pi + (self.m1 + self.m2) * self.g
        a = f_total / (self.m1 + self.m2)

        for _ in range(self.n_steps_per_action):
            # self.v1 = self.v1 + a * self.dt # simple Euler integration
            # self.y1 = self.y1 + self.v1 * self.dt

            v1_curr = self.v1
            y1_curr = self.y1
            v1_next = v1_curr + a * self.dt
            y1_next = y1_curr + v1_curr * self.dt  + 0.5 * a * self.dt**2 # mid-point Euler integration
            self.v1 = v1_next
            self.y1 = y1_next

        self.y1 = np.clip(self.y1, 0.0, self.pos_range)
        self.v1 = np.clip(self.v1, -self.half_vel_range, self.half_vel_range)

        y2 = self.y1 + self.l

        obs = np.array([self.y1, self.v1])

        reward = -self.reward_alpha * (y2 - self.h)**2 - self.reward_beta*f**2 - self.reward_gamma*self.v1**2

        done = False
        info = {}

        # self.step_cnt += 1
        # if self.step_cnt == 499:
        #     print()
        #     print('pi: ', pi)
        #     print('f: ', f)
        #     print('y2: ', y2)
        #     print()
        return obs, reward, done, info

    def reset(self):
        # self.v1 = 0.0
        # self.y1 = 0.0
        self.v1 = np.random.uniform(-self.half_vel_range, self.half_vel_range) # vel of both masses
        self.y1 = np.random.uniform(0, self.pos_range)
        self.step_cnt = 0
        return np.array([self.y1, self.v1])
    

    def render(self, mode='human'):
        pass