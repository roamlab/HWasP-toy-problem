'''
Parallel implementation of the Augmented Random Search method.

Adapted from code by:

Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''

import numpy as np
import time

import gym
from dowel import logger, tabular

from my_garage.algos import utils
from my_garage.algos.optimizers import SGD
from my_garage.algos.shared_noise import create_shared_noise
from my_garage.algos.shared_noise import SharedNoiseTable


class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_name='',
                 env=None,
                 policy_params = None,
                 policy=None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02,
                 discount=0.99):

        # initialize OpenAI environment for each worker
        if env is not None:
            self.env = env
        else:
            self.env = gym.make(env_name)
        self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params is None:
            self.policy = policy
        else:
            raise NotImplementedError
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length
        self.discount = discount


    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        return self.policy.get_weights_plus_stats()
    

    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward_list = []
        discounted_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            action, _ = self.policy.get_actions(np.array([ob]))
            ob, reward, done, _ = self.env.step(action[0])
            steps += 1
            total_reward_list.append(reward - shift)
            
            if done:
                break

        total_reward = np.sum(total_reward_list)
        gamma_list = [self.discount, ] * i
        discounted_reward = np.asarray(total_reward_list).dot(np.cumprod([1.] + gamma_list))
            
        return total_reward, discounted_reward, steps

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, discounted_rewards, deltas_idx = [], [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, discounted_reward, r_steps = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                rollout_rewards.append(reward)
                discounted_rewards.append(discounted_reward)
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                # self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.set_param_values(w_policy + delta)
                pos_reward, pos_discounted_reward, pos_steps  = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.set_param_values(w_policy - delta)
                neg_reward, neg_discounted_reward,  neg_steps = self.rollout(shift = shift) 
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                discounted_rewards.append([pos_discounted_reward, neg_discounted_reward])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, 'discounted_rewards': discounted_rewards, "steps" : steps}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return




class ARS(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 env=None, policy=None,
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=320, 
                 deltas_used=320,
                 delta_std=0.02, 
                 discount = 0.99,
                 logdir=None, 
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 seed=123,
                 ):


        if env is  None:
            env = gym.make(env_name)

        self.timesteps = 0
        self.action_size = env.action_space.shape[0]
        self.ob_size = env.observation_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')

        
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise()
        self.deltas = SharedNoiseTable(deltas_id, seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers

        self.workers = [Worker(seed + 7 * i,
                                      env_name=env_name,
                                      env=env,
                                      policy_params=policy_params,
                                      policy=policy,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std,
                                      discount=discount)
                        for i in range(num_workers)]

        # initialize policy 
        if policy_params is None:
            self.policy = policy
            self.w_policy = self.policy.get_param_values()

        else:
            raise NotImplementedError
            
        # initialize optimization algorithm
        self.optimizer = SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")


    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        # put policy weights in the object store
        # policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
            
        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts(self.w_policy,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts(self.w_policy,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results 
        results_one = rollout_ids_one
        results_two = rollout_ids_two

        rollout_rewards, discounted_rewards, deltas_idx = [], [], []

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            discounted_rewards += result['discounted_rewards']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            discounted_rewards += result['discounted_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        discounted_rewards = np.array(discounted_rewards, dtype = np.float64)

        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]
        tabular.record('AverageReturn', np.mean(rollout_rewards))
        tabular.record('MaxReturn', np.max(rollout_rewards))

        discounted_rewards = discounted_rewards[idx,:]
        tabular.record('AverageDiscountedReturn', np.mean(discounted_rewards))


        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat
        

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        
        g_hat = self.aggregate_rollouts(self.num_deltas)                    
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        return

    def train(self, num_iter, dump=False):

        start = time.time()
        for i in range(num_iter):
            with logger.prefix(' | Iteration {} |'.format(i)):
                t1 = time.time()
                self.train_step()
                t2 = time.time()
                print('total time of one step', t2 - t1)           
                print('iter ', i,' done')
                if dump:
                    logger.log(tabular)
                    logger.dump_all(i)
                    tabular.clear()
        return 