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

import numpy as np
import tensorflow as tf

from garage.tf.models.base import Model
from garage.tf.models.parameter import parameter
from garage.tf.models.mlp import mlp

from shared_params import params


class CompMechPolicyModel(Model):
    '''
    A Model only contains the structure/configuration of the underlying
    computation graphs.

    This comp. graph has input: obs y1 and v1
    and output: f, k, log_std
    '''
    def __init__(self, params, name='comp_mech_policy_model'):  # k_pre means pre-sigmoid
        super().__init__(name)
        self.k_pre_init = np.float32(params.k_pre_init)
        self.f_and_k_log_std_init = params.f_and_k_log_std_init

        self.pos_range = params.pos_range
        self.half_vel_range = params.half_vel_range
        self.k_pre_init_lb = params.k_pre_init_lb
        self.k_pre_init_ub = params.k_pre_init_ub
        self.comp_policy_network_size = params.comp_policy_network_size
        self.k_range = params.k_range
        self.k_lb = params.k_lb
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
            length=1,
            # initializer=tf.constant_initializer(self.k_pre_init),
            initializer=tf.random_uniform_initializer(minval=self.k_pre_init_lb, maxval=self.k_pre_init_ub),
            # initializer=tf.glorot_uniform_initializer(),
            trainable=True,
            name='k_pre')

        self.k_ts_normalized = tf.math.sigmoid(self.k_pre_var)

        y1_v1_k_ts_normalized = tf.concat([y1_and_v1_ph_normalized, self.k_ts_normalized], axis=1, name='y1_v1_k')

        f_ts_normalized = mlp(y1_v1_k_ts_normalized, 1, self.comp_policy_network_size, name='mlp', hidden_nonlinearity=tf.math.tanh, output_nonlinearity=tf.math.tanh)
        
        self.k_ts = tf.math.add(self.k_ts_normalized * tf.compat.v1.constant(self.k_range, dtype=tf.float32, name='k_range'), 
            tf.compat.v1.constant(self.k_lb, dtype=tf.float32, name='k_lb'), 
            name='k')

        self.f_ts = f_ts_normalized * self.half_force_range
        
        f_and_k_ts = tf.concat([self.f_ts, self.k_ts], axis = 1, name='f_and_k')

        self.log_std_var = parameter(
            input_var=y1_and_v1_ph,
            length=2,
            initializer=tf.constant_initializer(
                self.f_and_k_log_std_init),
            trainable=True,
            name='log_std')
        
        return f_and_k_ts, self.log_std_var


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
        return ['k', 'log_std']

    def get_mech_params(self):
        return [self.k_ts, self.log_std_var]


    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['k_pre_var']
        del new_dict['k_ts_normalized']
        del new_dict['k_ts']
        del new_dict['f_ts']
        del new_dict['log_std_var']
        return new_dict