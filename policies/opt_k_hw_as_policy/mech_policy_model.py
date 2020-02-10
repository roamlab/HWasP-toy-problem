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

from shared_params import params

k_ub = params.k_ub
k_lb = params.k_lb
k_range = params.k_range
# we use the sigmoid trick to limit the actual range of k: k = sigmoid(k_pre) * k_range + k_lb

half_force_range = params.half_force_range
k_pre_init_lb = params.k_pre_init_lb
k_pre_init_ub = params.k_pre_init_ub


class MechPolicyModel(Model):
    '''
    A Model only contains the structure/configuration of the underlying
    computation graphs.

    This comp. graph takes in the output of comp. policy f, and the state y1,
    and output the overall action pi, and additional information for reward calculation
    '''
    def __init__(self, params, name='mech_policy_model'):  # k_pre means pre-sigmoid
        super().__init__(name)
        self.k_pre_init = np.float32(params.k_pre_init)
        self.f_log_std_init = params.f_log_std_init

        self.half_force_range = params.half_force_range
        self.k_pre_init_lb = params.k_pre_init_lb
        self.k_pre_init_ub = params.k_pre_init_ub
        self.k_range = params.k_range
        self.k_lb = params.k_lb

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
        f_ph, y1_and_v1_ph = inputs[0] # f_ph: (?, 1), y1_and_v1_ph: (?, 2)
        f_ph = tf.multiply(f_ph, tf.compat.v1.constant(self.half_force_range, dtype=tf.float32, name='half_force_range')) # all these multiply() are scalar-tensor multiplication
        y1_ph = y1_and_v1_ph[:, 0:1]
        # self.k_pre_var = tf.compat.v1.get_variable('k_pre', initializer=self.k_pre_init, dtype=tf.float32, trainable=True)
        k_pre_init = np.random.uniform(self.k_pre_init_lb, self.k_pre_init_ub)
        self.k_pre_var = tf.compat.v1.get_variable(
            'k_pre', 
            initializer=k_pre_init, 
            dtype=tf.float32, 
            trainable=True)

        self.k_ts = tf.math.add(tf.nn.sigmoid(self.k_pre_var) * tf.compat.v1.constant(self.k_range, dtype=tf.float32, name='k_range'), 
            tf.compat.v1.constant(self.k_lb, dtype=tf.float32, name='k_lb'), 
            name='k')
        f_spring_ts = tf.multiply(-y1_ph, self.k_ts, name='f_spring')

        pi_ts = tf.add(f_ph, f_spring_ts, name='pi')


        self.log_std_var = parameter(
            input_var=y1_ph, # actually not linked to the input, this is just to match the dimension of the inputs for batches
            length=2,
            initializer=tf.constant_initializer(
                self.f_log_std_init),
            trainable=True,
            name='log_std')

        output_ts = tf.concat([pi_ts, f_ph], axis=1, name='pi_and_f')

        return output_ts, self.log_std_var # always see the combo (of pi and f) as the action

    def network_input_spec(self):
        """
        Network input spec.

        Return:
            *inputs (list[str]): List of key(str) for the network inputs.
        """
        return ['f', 'y1']

    def network_output_spec(self):
        """
        Network output spec.

        Return:
            *inputs (list[str]): List of key(str) for the network outputs.
        """
        return ['pi_and_f', 'log_std']

    def get_mech_params(self):
        return [self.k_pre_var, self.k_ts, self.log_std_var]


    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['k_pre_var']
        del new_dict['k_ts']
        del new_dict['log_std_var']
        return new_dict