'''
illustration of the system:

        --------------
        ^      \    
        |       \   
        |       /   spring, stiffness k masless, zero natural length 
        |      /    
        |      \    
        |       \    
        |    +-----+
        |    |     |
        |    |     |  point mass m1, position y1
 dist h |    +--+--+
        |       |
        |       | bar, devided into multiple segments, length l1, l2, ..., massless
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
from dowel import tabular


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def get_soft_conditioned_val(value1, value2, test_value, criterion, sigmoid_coeff=1.0):
    '''
    a soft way for the following if statement: (larger sigmoid_coeff means harder transition)

    if test_value < criterion:
        return value1
    else:
        return value2
    '''
    if value1 < value2:
        return sigmoid(sigmoid_coeff * (test_value - criterion)) * (value2 - value1) + value1
    else:
        return sigmoid(sigmoid_coeff * (-test_value + criterion)) * (value1 - value2) + value2


#################################### Base Class ####################################

class MassSpringEnv_OptL(gym.Env):
    '''
    1D mass-spring toy problem.
    Base class for Optimization Case II: optimizing the bar length l
    '''
    def __init__(self, params):
        # params
        self.half_force_range = params.half_force_range
        self.l_lb = params.l_lb
        self.l_ub = params.l_ub
        self.pos_range = params.pos_range
        self.half_vel_range = params.half_vel_range
        self.m1 = params.m1
        self.m2 = params.m2
        self.h = params.h
        self.g = params.g
        self.k = params.k
        self.dt = params.dt
        self.n_steps_per_action = params.n_steps_per_action
        self.n_steps_per_episode = params.n_steps_per_episode
        self.reward_alpha = params.reward_alpha
        self.reward_beta = params.reward_beta
        self.reward_gamma = params.reward_gamma
        self.reward_switch_pos_vel_thresh = params.reward_switch_pos_vel_thresh

        # states
        # self.v1 = np.random.uniform(-self.half_vel_range, self.half_vel_range) # vel of both masses
        # self.y1 = np.random.uniform(0, self.pos_range)
        self.y1 = 0.0
        self.v1 = 0.0
        self.step_cnt = 0

        # reward range
        # self.reward_range = (min([self.calc_reward(0.0, self.half_force_range, self.half_vel_range), self.calc_reward(self.pos_range, self.half_force_range, self.half_vel_range)]), 0.0)


    def step(self, action):
        raise NotImplementedError


    def simulate_w_naive_euler(self, y, v, a):
        for _ in range(self.n_steps_per_action):
            v = v + a * self.dt
            y = y + v * self.dt

        y = np.clip(y, 0.0, self.pos_range)
        v = np.clip(v, -self.half_vel_range, self.half_vel_range)
        return y, v


    def simulate_w_mid_point_euler(self, y, v, a):
        for _ in range(self.n_steps_per_action):
            v_curr = v
            y_curr = y
            v_next = v_curr + a * self.dt
            y_next = y_curr + v_curr * self.dt  + 0.5 * a * self.dt**2 # mid-point Euler integration
            v = v_next
            y = y_next
        y = np.clip(y, 0.0, self.pos_range)
        v = np.clip(v, -self.half_vel_range, self.half_vel_range)
        return y, v


    def calc_reward(self, y2, f, v2):
        pos_penalty = self.reward_alpha * np.abs(y2 - self.h)
        vel_penalty = self.reward_beta * np.abs(v2)

        force_penalty = get_soft_conditioned_val(self.reward_gamma * np.abs(f), self.reward_gamma * np.abs(self.half_force_range), pos_penalty + vel_penalty, self.reward_switch_pos_vel_thresh, 50.0)
        
        reward = -pos_penalty - vel_penalty - force_penalty
        return reward


    def reset(self):
        raise NotImplementedError
    

    def render(self, mode='human'):
        pass



#################################### Hardware as Action ####################################


class MassSpringEnv_OptL_HwAsAction(MassSpringEnv_OptL):
    '''
    Action: f, l1, l2, ...
    observation: y1, v1
    '''

    def __init__(self, params):
        super().__init__(params)
        self.n_segments = params.n_segments
        # action space, different for different subclasses
        l_lb_list = [self.l_lb,] * self.n_segments
        l_ub_list = [self.l_ub,] * self.n_segments

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -self.half_vel_range]), 
            high=np.array([self.pos_range, self.half_vel_range]), 
            dtype=np.float32) # obs y1 and v1

        self.action_space = gym.spaces.Box(
            low=np.array([-self.half_force_range] + l_lb_list), 
            high=np.array([self.half_force_range] + l_ub_list), 
            dtype=np.float32) # 1st: redifined action pi, 2nd: original action f


    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the policy, here a combination of f and l1, l2, ...
        
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self.step_cnt += 1
        action = np.clip(action.copy(), self.action_space.low, self.action_space.high)
        f = action[0] # input force
        l = np.sum(action[1:]) # bar length
        f_total = f + (self.m1 + self.m2) * self.g - self.k * self.y1
        a = f_total / (self.m1 + self.m2)
        self.y1, self.v1 = self.simulate_w_mid_point_euler(self.y1, self.v1, a)
        y2 = self.y1 + l
        obs = np.array([self.y1, self.v1])
        reward = self.calc_reward(y2, f, self.v1)
        done = False
        info = {}
        if self.step_cnt == self.n_steps_per_episode:
            print()
            print('y2: ', y2)
            print('v2: ', self.v1)
            print('l: ', l)
            tabular.record('Env/FinalL', l)

        return obs, reward, done, info


    def reset(self):
        self.v1 = np.random.uniform(-self.half_vel_range, self.half_vel_range) # vel of both masses
        self.y1 = np.random.uniform(0, self.pos_range)
        # self.y1 = 0.0
        # self.v1 = 0.0

        self.step_cnt = 0
        return np.array([self.y1, self.v1])



#################################### Hardware as Policy ####################################



class MassSpringEnv_OptL_HwAsPolicy(MassSpringEnv_OptL):
    def __init__(self, params):
        super().__init__(params)
 
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -self.half_vel_range, 0.0, -self.half_vel_range]), 
            high=np.array([self.pos_range, self.half_vel_range, self.pos_range, self.half_vel_range]), 
            dtype=np.float32) # obs y1 v1 y2 v2, but only y1 v1 are used for computational policy


        self.action_space = gym.spaces.Box(
            low=-self.half_force_range, 
            high=self.half_force_range, 
            shape=(3, ), 
            dtype=np.float32) # 1st: interface force on m1, 2nd: interface force on m2, 3rd: original action f


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
        self.step_cnt += 1
        action = np.clip(action.copy(), self.action_space.low, self.action_space.high)
        f1 = action[0]  # interface force on m1
        f2 = action[1]  # interface force on m2
        f = action[2]   # original action f
        f_total_1 = f1 + self.m1 * self.g - self.k * self.y1
        a1 = f_total_1 / self.m1
        f_total_2 = f2 + f + self.m2 * self.g
        a2 = f_total_2 / self.m2

        self.y1, self.v1 = self.simulate_w_mid_point_euler(self.y1, self.v1, a1)
        self.y2, self.v2 = self.simulate_w_mid_point_euler(self.y2, self.v2, a2)

        obs = np.array([self.y1, self.v1, self.y2, self.v2])
        reward = self.calc_reward(self.y2, f, self.v1)
        done = False
        info = {}
        if self.step_cnt == self.n_steps_per_episode:
            print()
            print('y2: ', self.y2)
            print('v2: ', self.v2)
        return obs, reward, done, info


    def reset(self):
        self.v1 = np.random.uniform(-self.half_vel_range, self.half_vel_range) # vel of both masses
        self.y1 = np.random.uniform(0, self.pos_range)
        # self.y1 = 0.0
        # self.v1 = 0.0

        l_avg = 1/2 * (self.l_lb + self.l_ub)
        self.v2 = self.v1 # just a guess, will be recalculated
        self.y2 = self.y1 + l_avg # just a guess, will be recalculated

        self.step_cnt = 0
        return np.array([self.y1, self.v1, self.y2, self.v2])