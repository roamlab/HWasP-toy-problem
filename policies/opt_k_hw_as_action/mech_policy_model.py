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


class MechPolicyModel(Model):
    '''
    A Model only contains the structure/configuration of the underlying
    computation graphs.

    This comp. graph do not have inputs, 
    and output the overall action pi, and additional information for reward calculation
    '''
    def __init__(self, k_pre_init, log_std_init, name='mech_policy_model'):  # k_pre means pre-sigmoid
        super().__init__(name)
        self.k_pre_init = np.float32(k_pre_init)
        self.log_std_init = log_std_init

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
            length=1,
            initializer=tf.constant_initializer(
                self.k_pre_init),
            trainable=True,
            name='k_pre')

        self.k_ts = tf.math.add(tf.math.sigmoid(self.k_pre_var) * tf.compat.v1.constant(k_range, dtype=tf.float32, name='k_range'), 
            tf.compat.v1.constant(k_lb, dtype=tf.float32, name='k_lb'), 
            name='k')

        self.log_std_var = parameter(
            input_var=inputs[0],
            length=2,
            initializer=tf.constant_initializer(
                self.log_std_init),
            trainable=True,
            name='log_std')
        
        return self.k_ts, self.log_std_var


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
        return [self.k_pre_var, self.k_ts, self.log_std_var]


    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['k_pre_var']
        del new_dict['k_ts']
        del new_dict['log_std_var']
        return new_dict