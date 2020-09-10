'''
illustration of the system:

        --------------
        ^    \    \
        |     \    \
        |     / .. /  springs, stiffness k1, k2 ..., masless, zero natural length 
        |    /    /
        |    \    \
        |     \    \
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

import numpy as np
import tensorflow as tf

from garage.tf.models.base import Model
from garage.tf.models.parameter import parameter
from garage.tf.models.mlp import mlp


#################################### Base Class ####################################


class MyBaseModel_OptK(Model):
    '''
    A Model only contains the structure/configuration of the underlying
    computation graphs.
    '''
    def __init__(self, params, name='my_base_model'):
        super().__init__(name)
        
        self.k_pre_init = np.float32(params.k_pre_init) # k_pre means pre-sigmoid
        self.k_pre_init_lb = params.k_pre_init_lb
        self.k_pre_init_ub = params.k_pre_init_ub
        self.k_range = params.k_range
        self.k_lb = params.k_lb
        self.n_springs = params.n_springs
    
        self.trq_const = params.trq_const
        self.r_shaft = params.r_shaft

    def network_input_spec(self):
        """
        Network input spec.

        Return:
            *inputs (list[str]): List of key(str) for the network inputs.
        """
        return ['y1_and_v1']


    def network_output_spec(self):
        """
        Network output spec.

        Return:
            *inputs (list[str]): List of key(str) for the network outputs.
        """
        raise NotImplementedError      


    def _build(self, *inputs, name=None):
        raise NotImplementedError


#################################### Hardware as Action ####################################


class MechPolicyModel_OptK_HwAsAction(MyBaseModel_OptK):
    def __init__(self, params, name='mech_policy_model'):
        super().__init__(params, name=name)
        self.f_and_k_log_std_init = [params.f_log_std_init_action,] + [params.k_log_std_init_action,] * params.n_springs

    def _build(self, *inputs, name=None):
        """
        Output of the model given input placeholder(s).

        User should implement _build() inside their subclassed model,
        and construct the computation graphs in this function.

        Args:
            inputs: Tensor input(s), recommended to be position arguments, e.g.
              def _build(self, state_input, action_input, name=None).
              It would be usually same as the inputs in build().
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Return:
            output: Tensor output(s) of the model.
        """

        # the inputs are y1_and_v1_ph
        # the input values are not used, only the dimensions are used

        self.k_pre_var = parameter(
            input_var=inputs[0],
            length=self.n_springs,
            # initializer=tf.constant_initializer(self.k_pre_init), # uncomment this line when training cmaes(hyperparameter)+ppo
            # trainable=False,  # uncomment this line when training cmaes(hyperparameter)+ppo

            initializer=tf.random_uniform_initializer(minval=self.k_pre_init_lb, maxval=self.k_pre_init_ub), # uncomment this line when training ppo only
            trainable=True, # uncomment this line when training ppo only
            name='k_pre')

        self.k_ts = tf.math.add(tf.math.sigmoid(self.k_pre_var) * tf.compat.v1.constant(self.k_range, dtype=tf.float32, name='k_range'), 
            tf.compat.v1.constant(self.k_lb, dtype=tf.float32, name='k_lb'), 
            name='k')

        # the mean in the output of this model only contains k's,but log_std contains the stds for f and k's
        self.log_std_var = parameter(
            input_var=inputs[0],
            length=1+self.n_springs,
            initializer=tf.constant_initializer(
                self.f_and_k_log_std_init),
            trainable=True,
            name='log_std')
        return self.k_ts, self.log_std_var


    def network_output_spec(self):
        """
        Network output spec.

        Return:
            *inputs (list[str]): List of key(str) for the network outputs.
        """
        return ['k', 'log_std']


    def get_tensors(self):
        return dict(k_pre_var=self.k_pre_var, 
                    k_ts=self.k_ts, 
                    k_sum_ts=self.k_sum_ts, 
                    log_std_var=self.log_std_var)


    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['k_pre_var']
        del new_dict['k_ts']
        del new_dict['log_std_var']
        return new_dict


#################################### Hardware as Policy ####################################


class MechPolicyModel_OptK_HwAsPolicy(MyBaseModel_OptK):
    def __init__(self, params, name='mech_policy_model'):
        super().__init__(params, name=name)

        self.pi_and_f_log_std_init = [params.f_log_std_init_action, params.f_log_std_init_auxiliary]
        self.half_force_range = params.half_force_range


    def _build(self, *inputs, name=None):
        """
        Output of the model given input placeholder(s).

        User should implement _build() inside their subclassed model,
        and construct the computation graphs in this function.

        Args:
            inputs: Tensor input(s), recommended to be position arguments, e.g.
              def _build(self, state_input, action_input, name=None).
              It would be usually same as the inputs in build().
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Return:
            output: Tensor output(s) of the model.
        """
        i_ph_normalized, y1_and_v1_ph = inputs[0] # i_ph_normalized: (?, 1), y1_and_v1_ph: (?, 2)
        f_ts_normalized = i_ph_normalized * self.trq_const / self.r_shaft
        f_ts = tf.multiply(f_ph_normalized[:, 0], tf.compat.v1.constant(self.half_force_range, dtype=tf.float32, name='half_force_range'), name='f') # scalar-tensor multiplication # f_ts: (?,)
        y1_ph = y1_and_v1_ph[:, 0] # y1_ph: (?,) 
        # self.k_pre_var = tf.compat.v1.get_variable('k_pre', initializer=[self.k_pre_init,] * self.n_springs, dtype=tf.float32, trainable=True)
        k_pre_init = np.float32(np.random.uniform(self.k_pre_init_lb, self.k_pre_init_ub, size=(self.n_springs,)))
        self.k_pre_var = tf.compat.v1.get_variable(
            'k_pre', 
            initializer=k_pre_init, 
            dtype=tf.float32, 
            trainable=True)

        self.k_ts = tf.math.add(tf.nn.sigmoid(self.k_pre_var) * tf.compat.v1.constant(self.k_range, dtype=tf.float32, name='k_range'), 
            tf.compat.v1.constant(self.k_lb, dtype=tf.float32, name='k_lb'), name='k')
        self.k_sum_ts =  tf.math.reduce_sum(self.k_ts) # only for monitoring the k

        y1_mat =  tf.transpose(tf.tile([y1_ph], [self.n_springs, 1]), name='y1_mat') 
        # y1_mat: (?, self.n_springs), [[y1[1], y1[1], ...], ...[y1[?], y1[?], ...]]
        f_spring_ts = -tf.linalg.matvec(y1_mat, self.k_ts, name='f_spring')
        #  f_spring_ts: (?,), -[y1[1]*k[1]+y1[1]*k[2]+... , ... , y1[?]*k[1]+y1[?]*k[2]+...]
        pi_ts = tf.add(f_ts, f_spring_ts, name='pi') # pi_ts (?,)

        # f_ts_stop_grad = tf.compat.v1.stop_gradient(f_ts) # we should not stop gradient, but should see k as an actual action with the ability to backprop

        # pi_and_f_ts = tf.concat([tf.expand_dims(pi_ts, axis=-1), tf.expand_dims(f_ts, axis=-1)], axis=1) 
        pi_and_f_ts = tf.stack([pi_ts, f_ts], axis=1, name='pi_and_f') # pi_and_f_ts: (?, 2)

        self.debug_ts = tf.gradients(tf.log(pi_and_f_ts), self.k_pre_var)

        self.log_std_var = parameter(
            input_var=y1_and_v1_ph, # actually not linked to the input, this is just to match the dimension of the inputs for batches
            length=2,
            initializer=tf.constant_initializer(
                self.pi_and_f_log_std_init),
            trainable=True,
            name='log_std') 
            # shape: (?, 2)

        return pi_and_f_ts, self.log_std_var # always see the combo (of pi and f) as the action


    def network_output_spec(self):
        """
        Network output spec.

        Return:
            *inputs (list[str]): List of key(str) for the network outputs.
        """
        return ['pi_and_f', 'log_std']


    def get_tensors(self):
        return dict(k_pre_var=self.k_pre_var, 
                    k_ts=self.k_ts, 
                    k_sum_ts=self.k_sum_ts, 
                    log_std_var=self.log_std_var,
                    debug_ts=self.debug_ts)


    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['k_pre_var']
        del new_dict['k_ts']
        del new_dict['k_sum_ts']
        del new_dict['log_std_var']
        del new_dict['debug_ts']
        return new_dict


################################### Hardware in Policy and Action ###################################

class CompMechPolicyModel_OptK_HwInPolicyAndAction(MyBaseModel_OptK):
    def __init__(self, params, name='comp_mech_policy_model'):
        super().__init__(params, name=name)
        from garage.tf.models.mlp import mlp

        self.f_and_k_log_std_init = [params.f_log_std_init_action,] +  [params.k_log_std_init_auxiliary,] * self.n_springs
        self.pos_range = params.pos_range
        self.half_vel_range = params.half_vel_range
        self.comp_policy_network_size = params.comp_policy_network_size
        self.half_force_range = params.half_force_range


    def _build(self, *inputs, name=None):
        """
        Output of the model given input placeholder(s).

        User should implement _build() inside their subclassed model,
        and construct the computation graphs in this function.

        Args:
            inputs: Tensor input(s), recommended to be position arguments, e.g.
              def _build(self, state_input, action_input, name=None).
              It would be usually same as the inputs in build().
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Return:
            output: Tensor output(s) of the model.
        """

        # the inputs are y1_and_v1_ph
        y1_and_v1_ph = inputs[0]

        y1_and_v1_ph_normalized = y1_and_v1_ph / [self.pos_range, self.half_vel_range]

        self.k_pre_var = parameter(
            input_var=y1_and_v1_ph,
            length=self.n_springs,
            # initializer=tf.constant_initializer(self.k_pre_init),
            initializer=tf.random_uniform_initializer(minval=self.k_pre_init_lb, maxval=self.k_pre_init_ub),
            # initializer=tf.glorot_uniform_initializer(),
            trainable=True,
            name='k_pre')

        self.k_ts_normalized = tf.math.sigmoid(self.k_pre_var)

        y1_v1_k_ts_normalized = tf.concat([y1_and_v1_ph_normalized, self.k_ts_normalized], axis=1, name='y1_v1_k')

        f_ts_normalized = mlp(y1_v1_k_ts_normalized, 1, self.comp_policy_network_size, name='mlp', hidden_nonlinearity=tf.math.tanh, output_nonlinearity=tf.math.tanh)
        
        self.f_ts = f_ts_normalized * self.half_force_range

        self.k_ts = tf.math.add(self.k_ts_normalized * tf.compat.v1.constant(self.k_range, dtype=tf.float32, name='k_range'), 
            tf.compat.v1.constant(self.k_lb, dtype=tf.float32, name='k_lb'), 
            name='k')

        # k_ts_stop_grad = tf.stop_gradient(self.k_ts) # we should not stop gradient, but should see k as an actual action,

        f_and_k_ts = tf.concat([self.f_ts, self.k_ts], axis = 1, name='f_and_k')

        self.debug_ts = tf.gradients(f_and_k_ts, self.k_pre_var)

        self.log_std_var = parameter(
            input_var=y1_and_v1_ph,
            length=1+self.n_springs,
            initializer=tf.constant_initializer(
                self.f_and_k_log_std_init),
            trainable=True,
            name='log_std')
        
        return f_and_k_ts, self.log_std_var


    def network_output_spec(self):
        """
        Network output spec.

        Return:
            *inputs (list[str]): List of key(str) for the network outputs.
        """
        return ['f_and_k', 'log_std']


    def get_tensors(self):
        return dict(k_ts=self.k_ts, 
                    log_std_var=self.log_std_var,
                    debug_ts=self.debug_ts)


    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['k_pre_var']
        del new_dict['k_ts_normalized']
        del new_dict['k_ts']
        del new_dict['f_ts']
        del new_dict['debug_ts']
        del new_dict['log_std_var']
        return new_dict