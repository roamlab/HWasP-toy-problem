import numpy as np
import tensorflow as tf

from garage.tf.policies.base import StochasticPolicy
from garage.tf.distributions.diagonal_gaussian import DiagonalGaussian

from shared_params import params

#################################### Base Class ####################################

class MyBasePolicy(StochasticPolicy):
    def __init__(self, env_spec, name='my_base_policy'):
        super().__init__(env_spec=env_spec, name=name)
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self._dist = DiagonalGaussian(dim=self.action_dim)


    def _initialize(self):
        # subclasses need to define the callable self._f_policy
        raise NotImplementedError


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
        means, log_stds = self._f_policy(flat_obs)
        rnd = np.random.normal(size=means.shape)
        samples = rnd * np.exp(log_stds) + means
        samples = self.action_space.unflatten_n(samples)
        means = self.action_space.unflatten_n(means)
        log_stds = self.action_space.unflatten_n(log_stds)
        info = dict(mean=means, log_std=log_stds)
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
        mean, log_std = self._f_policy([flat_obs])
        rnd = np.random.normal(size=mean.shape)
        sample = rnd * np.exp(log_std) + mean
        sample = self.action_space.unflatten(sample[0])
        mean = self.action_space.unflatten(mean[0])
        log_std = self.action_space.unflatten(log_std[0])
        info = dict(mean=mean, log_std=log_std)
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
        raise NotImplementedError


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


#################################### Hardware as Action ####################################


class CompMechPolicy_OptK_HwAsAction(MyBasePolicy):
    def __init__(self, env_spec,
                comp_policy_model, 
                mech_policy_model, 
                name='comp_mech_policy'):
        super().__init__(env_spec=env_spec, name=name)
        self.comp_policy_model = comp_policy_model
        self.mech_policy_model = mech_policy_model
        self._initialize()


    def _initialize(self):
        y1_and_v1_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, self.obs_dim), name='y1_and_v1_ph') # obs: y1 and v1
        y1_and_v1_ph_normalized = y1_and_v1_ph / [params.pos_range, params.half_vel_range]
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            f_ts = self.comp_policy_model.build(y1_and_v1_ph_normalized) * params.half_force_range
            k_ts, log_std_ts = self.mech_policy_model.build(y1_and_v1_ph_normalized)

            f_and_k_ts = tf.concat([f_ts, k_ts], axis=1, name='action')

        self._f_policy = tf.compat.v1.get_default_session().make_callable([f_and_k_ts, log_std_ts], feed_list=[y1_and_v1_ph])


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
        obs_var_normalized = obs_var / [params.pos_range, params.half_vel_range]

        with tf.compat.v1.variable_scope(self._variable_scope):
            f_ts = self.comp_policy_model.build(obs_var_normalized, name=name) * params.half_force_range
            k_ts, log_std_ts = self.mech_policy_model.build(obs_var_normalized, name=name)

            f_and_k_ts = tf.concat([f_ts, k_ts], axis=1, name='action')

        return dict(
            mean = f_and_k_ts,
            log_std = log_std_ts
        )


#################################### Hardware as Policy ####################################


class CompMechPolicy_OptK_HwAsPolicy(MyBasePolicy):
    def __init__(self, env_spec,
                comp_policy_model, 
                mech_policy_model, 
                name='comp_mech_policy'):
        super().__init__(env_spec=env_spec, name=name)
        self.comp_policy_model = comp_policy_model
        self.mech_policy_model = mech_policy_model
        self._initialize()


    def _initialize(self):
        y1_and_v1_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, self.obs_dim), name='y1_and_v1_ph') # obs: y1 and v1
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            f_ts = self.comp_policy_model.build(y1_and_v1_ph)
            pi_and_f_ts, log_std_ts = self.mech_policy_model.build([f_ts, y1_and_v1_ph])

        self._f_policy = tf.compat.v1.get_default_session().make_callable([pi_and_f_ts, log_std_ts], feed_list=[y1_and_v1_ph])


    def get_actions(self, observations):
        samples, info = super().get_actions(observations)
        k_ts = self.mech_policy_model.get_mech_params()[1]
        k = np.full(samples.shape, tf.compat.v1.get_default_session().run(k_ts))
        info['k'] = k
        return samples, info
    

    def get_action(self, observation):
        sample, info = super().get_action(observation)
        k_ts = self.mech_policy_model.get_mech_params()[1]
        k = np.full(sample.shape, tf.compat.v1.get_default_session().run(k_ts))
        info['k'] = k
        return sample, info


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


#################################### Hardware in Policy and Action ####################################


class CompMechPolicy_OptK_HwInPolicyAndAction(MyBasePolicy):
    def __init__(self, 
                env_spec,
                comp_mech_policy_model,
                name='comp_mech_policy'
                ):
        super().__init__(env_spec=env_spec, name=name)
        self.comp_mech_policy_model = comp_mech_policy_model
        self._initialize()

    def _initialize(self):
        y1_and_v1_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, self.obs_dim), name='y1_and_v1_ph') # obs: y1 and v1
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            f_and_k_ts, log_std_ts = self.comp_mech_policy_model.build(y1_and_v1_ph)

        self._f_policy = tf.compat.v1.get_default_session().make_callable([f_and_k_ts, log_std_ts], feed_list=[y1_and_v1_ph])


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
            f_and_k_ts, log_std_ts = self.comp_mech_policy_model.build(obs_var, name=name)


        return dict(
            mean = f_and_k_ts,
            log_std = log_std_ts
        )