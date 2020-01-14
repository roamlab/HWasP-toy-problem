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
from garage.tf.core.parameter import parameter


K_UB = 1e2
K_LB = 0
K_RANGE = K_UB - K_LB
# we use the sigmoid trick to limit the actual range of k: k = sigmoid(k_pre) * K_RANGE + K_LB
# TODO: add scaling if there are numerical issues


class MechPolicyModel(Model):
    '''
    A Model only contains the structure/configuration of the underlying
    computation graphs.

    This comp. graph takes in the output of comp. policy f, and the state y1,
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
        f_ph, y1_ph = inputs[0]
        self.k_pre_var = tf.compat.v1.get_variable('k_pre', initializer=self.k_pre_init, dtype=tf.float32, trainable=True)
        self.k_ts = tf.math.add(tf.nn.sigmoid(self.k_pre_var) * tf.compat.v1.constant(K_RANGE, dtype=tf.float32, name='K_RANGE'), 
            tf.compat.v1.constant(K_LB, dtype=tf.float32, name='K_LB'), 
            name='k')
        f_spring_ts = tf.multiply(-y1_ph, self.k_ts, name='f_spring')

        pi_ts = tf.add(f_ph, f_spring_ts, name='pi')


        self.log_std_var = parameter(
            input_var=y1_ph,
            length=2,
            initializer=tf.constant_initializer(
                self.log_std_init),
            trainable=False,
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