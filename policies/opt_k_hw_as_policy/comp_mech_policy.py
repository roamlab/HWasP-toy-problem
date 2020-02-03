'''The overall policy consisting of the computational policy and the mechanical policy'''
import numpy as np
import tensorflow as tf

from garage.tf.policies.base import StochasticPolicy
from garage.tf.distributions.diagonal_gaussian import DiagonalGaussian

from policies.opt_k_hw_as_policy.mech_policy_model import MechPolicyModel


class CompMechPolicy_OptK_HwAsPolicy(StochasticPolicy):
    def __init__(self, 
                env_spec,
                comp_policy_model,
                mech_policy_model,
                name='comp_mech_policy_opt_k_hw_as_policy'
                ):
        super().__init__(name, env_spec)
        self.comp_policy_model = comp_policy_model
        self.mech_policy_model = mech_policy_model
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self._initialize()


    def _initialize(self):
        y1_and_v1_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, self.obs_dim), name='y1_and_v1_ph') # obs: y1 and v1
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            f_ts = self.comp_policy_model.build(y1_and_v1_ph)
            pi_and_f_ts, log_std_ts = self.mech_policy_model.build([f_ts, y1_and_v1_ph])

        self._f_policy = tf.compat.v1.get_default_session().make_callable([pi_and_f_ts, log_std_ts], feed_list=[y1_and_v1_ph])
        self._dist = DiagonalGaussian(dim=self.action_dim)



    def get_actions(self, observations):
        '''
        Get multiple actions from this policy for the input observations.
        Args:
            observations (numpy.ndarray): Observations from environment.
        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information.
        Note:
            It returns actions and a dict, with keys
            - mean (numpy.ndarray): Means of the distribution.
            - log_std (numpy.ndarray): Log standard deviations of the
                distribution.
        '''

        flat_obs = self.observation_space.flatten_n(observations)
        pi_and_f, log_stds = self._f_policy(flat_obs)
        rnd = np.random.normal(size=pi_and_f.shape)
        pi_and_f_samples = rnd * np.exp(log_stds) + pi_and_f
        samples = self.action_space.unflatten_n(pi_and_f_samples)
        mean = self.action_space.unflatten_n(pi_and_f)
        log_stds = self.action_space.unflatten_n(log_stds)
        k_ts = self.mech_policy_model.get_mech_params()[1]
        k = np.full(mean.shape, tf.compat.v1.get_default_session().run(k_ts))
        info = dict(mean=mean, log_std=log_stds, k=k)
        return samples, info


    def get_action(self, observation):
        '''
        Get single action from this policy for the input observation.
        Args:
            observation (numpy.ndarray): Observation from environment.
        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information.
        Note:
            It returns an action and a dict, with keys
            - mean (numpy.ndarray): Mean of the distribution.
            - log_std (numpy.ndarray): Log standard deviation of the
                distribution.
        '''
        flat_obs = self.observation_space.flatten(observation)
        pi_and_f, log_std = self._f_policy([flat_obs])
        rnd = np.random.normal(size=pi_and_f.shape)
        pi_and_f_sample = rnd * np.exp(log_std) + pi_and_f
        sample = self.action_space.unflatten(pi_and_f_sample[0])
        mean = self.action_space.unflatten(pi_and_f[0])
        log_std = self.action_space.unflatten(log_std[0])
        k_ts = self.mech_policy_model.get_mech_params()[1]
        k = np.full(mean.shape, tf.compat.v1.get_default_session().run(k_ts))
        info = dict(mean=mean, log_std=log_std, k=k)
        return sample, info

    @property
    def distribution(self):
        return self._dist

    def dist_info_sym(self, obs_var, state_info_vars, name='default'):
        """
        Symbolic graph of the distribution.

        Return the symbolic distribution information about the actions.
        Args:
            obs_var (tf.Tensor): symbolic variable for observations
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation.
            name (str): Name of the symbolic graph.

        :return:
        """

        with tf.compat.v1.variable_scope(self._variable_scope):
            f_ts = self.comp_policy_model.build(obs_var, name=name)
            pi_and_f_ts, log_std_ts = self.mech_policy_model.build([f_ts, obs_var], name=name)

        return dict(
            mean = pi_and_f_ts,
            log_std = log_std_ts
        )



    def dist_info(self, obs, state_infos):
        """
        Distribution info.

        Return the distribution information about the actions.

        Args:
            obs_var (tf.Tensor): observation values
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation
        """
        raise NotImplementedError

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['_f_policy']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        super().__setstate__(state)
        self._initialize()
    
    @property
    def vectorized(self):
        """Vectorized or not."""
        return True