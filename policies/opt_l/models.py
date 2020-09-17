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

import numpy as np
import tensorflow as tf

from garage.tf.models.base import Model
from garage.tf.models.parameter import parameter
from garage.tf.models.mlp import mlp


#################################### Base Class ####################################


class MyBaseModel_OptL(Model):
    '''
    A Model only contains the structure/configuration of the underlying
    computation graphs.
    '''
    def __init__(self, params, name='my_base_model'):
        super().__init__(name)
        
        self.l_pre_init = np.float32(params.l_pre_init) # l_pre means pre-sigmoid
        self.l_pre_init_lb = params.l_pre_init_lb
        self.l_pre_init_ub = params.l_pre_init_ub
        self.l_range = params.l_range
        self.l_lb = params.l_lb
        self.n_segments = params.n_segments
    

    def network_input_spec(self):
        """
        Network input spec.

        Return:
            *inputs (list[str]): List of key(str) for the network inputs.
        """
        raise NotImplementedError      


    def network_output_spec(self):
        """
        Network output spec.

        Return:
            *inputs (list[str]): List of key(str) for the network outputs.
        """
        raise NotImplementedError      


    def _build(self, *inputs, name=None):
        raise NotImplementedError


#################################### Fixed Hardware ####################################


class MechPolicyModel_OptL_FixedHW(MyBaseModel_OptL):
    def __init__(self, params, name='mech_policy_model'):
        super().__init__(params, name=name)
        self.f_and_l_log_std_init = [params.f_log_std_init_action,] + [params.l_log_std_init_action,] * params.n_segments

    def _build(self, inputs, name=None):
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

        self.l_pre_var = parameter(
            input_var=inputs,
            length=self.n_segments,
            initializer=tf.constant_initializer(self.l_pre_init), 
            trainable=False,  
            name='l_pre')

        self.l_ts = tf.math.add(tf.math.sigmoid(self.l_pre_var) * tf.compat.v1.constant(self.l_range, dtype=tf.float32, name='l_range'), 
            tf.compat.v1.constant(self.l_lb, dtype=tf.float32, name='l_lb'), 
            name='l')

        # the mean in the output of this model only contains l's,but log_std contains the stds for f and l's
        self.log_std_var = parameter(
            input_var=inputs,
            length=1+self.n_segments,
            initializer=tf.constant_initializer(
                self.f_and_l_log_std_init),
            trainable=True,
            name='log_std')
        return self.l_ts, self.log_std_var


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
        return ['l', 'log_std']


    def get_tensors(self):
        return dict(l_pre_var=self.l_pre_var, 
                    l_ts=self.l_ts, 
                    l_sum_ts=self.l_sum_ts, 
                    log_std_var=self.log_std_var)


    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['l_pre_var']
        del new_dict['l_ts']
        del new_dict['log_std_var']
        return new_dict


#################################### Hardware as Action ####################################


class MechPolicyModel_OptL_HwAsAction(MyBaseModel_OptL):
    def __init__(self, params, name='mech_policy_model'):
        super().__init__(params, name=name)
        self.f_and_l_log_std_init = [params.f_log_std_init_action,] + [params.l_log_std_init_action,] * params.n_segments

    def _build(self, inputs, name=None):
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

        self.l_pre_var = parameter(
            input_var=inputs,
            length=self.n_segments,
            initializer=tf.random_uniform_initializer(minval=self.l_pre_init_lb, maxval=self.l_pre_init_ub),
            trainable=True,
            name='l_pre')

        self.l_ts = tf.math.add(tf.math.sigmoid(self.l_pre_var) * tf.compat.v1.constant(self.l_range, dtype=tf.float32, name='l_range'), 
            tf.compat.v1.constant(self.l_lb, dtype=tf.float32, name='l_lb'), 
            name='l')

        # the mean in the output of this model only contains l's,but log_std contains the stds for f and l's
        self.log_std_var = parameter(
            input_var=inputs,
            length=1+self.n_segments,
            initializer=tf.constant_initializer(
                self.f_and_l_log_std_init),
            trainable=True,
            name='log_std')
        return self.l_ts, self.log_std_var


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
        return ['l', 'log_std']


    def get_tensors(self):
        return dict(l_pre_var=self.l_pre_var, 
                    l_ts=self.l_ts, 
                    l_sum_ts=self.l_sum_ts, 
                    log_std_var=self.log_std_var)


    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['l_pre_var']
        del new_dict['l_ts']
        del new_dict['log_std_var']
        return new_dict


#################################### Hardware as Policy ####################################


class MechPolicyModel_OptL_HwAsPolicy(MyBaseModel_OptL):
    def __init__(self, params, name='mech_policy_model'):
        super().__init__(params, name=name)

        self.f1_f2_f_log_std_init = [params.f_log_std_init_action, params.f_log_std_init_action, params.f_log_std_init_auxiliary]
        self.half_force_range = params.half_force_range
        self.k_interface = params.k_interface
        self.b_interface = params.b_interface


    def _build(self, inputs, name=None):
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
        f_ph_normalized, y1_v1_y2_v2_ph = inputs # f_ph_normalized: (?, 1), y1_v1_y2_v2_ph: (?, 4)

        f_ts = tf.multiply(f_ph_normalized[:, 0], tf.compat.v1.constant(self.half_force_range, dtype=tf.float32, name='half_force_range'), name='f') # scalar-tensor multiplication # f_ts: (?,)

        y1_ph = y1_v1_y2_v2_ph[:, 0] # y1_ph: (?,) 
        v1_ph = y1_v1_y2_v2_ph[:, 1] # v1_ph: (?,) 
        y2_ph = y1_v1_y2_v2_ph[:, 2] # y2_ph: (?,) 
        v2_ph = y1_v1_y2_v2_ph[:, 3] # v2_ph: (?,) 

        l_pre_var = parameter(
            input_var=y1_v1_y2_v2_ph,
            length=self.n_segments,
            # initializer=tf.constant_initializer(self.l_pre_init),
            initializer=tf.random_uniform_initializer(minval=self.l_pre_init_lb, maxval=self.l_pre_init_ub),
            trainable=True,
            name='l_pre')

        l_segment_ts = tf.math.add(tf.math.sigmoid(l_pre_var) * tf.compat.v1.constant(self.l_range, dtype=tf.float32, name='l_range'), 
            tf.compat.v1.constant(self.l_lb, dtype=tf.float32, name='l_lb'), 
            name='l')
        
        self.l_ts = tf.math.reduce_sum(l_segment_ts, axis=-1)

        f1_ts = 0.5 * self.k_interface * (y2_ph - y1_ph - self.l_ts) + 0.5 * self.b_interface * (v2_ph - v1_ph) # see the notes for the derivation
        f2_ts = -f1_ts # the bar has no mass 

        f1_f2_f_ts = tf.stack([f1_ts, f2_ts, f_ts], axis=1, name='f1_f2_f')

        self.debug_ts = self.l_ts

        log_std_var = parameter(
            input_var=y1_v1_y2_v2_ph, # actually not linked to the input, this is just to match the dimension of the inputs for batches
            length=3,
            initializer=tf.constant_initializer(self.f1_f2_f_log_std_init),
            trainable=True,
            name='log_std') 
            # shape: (?, 3)

        return f1_f2_f_ts, log_std_var


    def network_input_spec(self):
        """
        Network input spec.

        Return:
            *inputs (list[str]): List of key(str) for the network inputs.
        """
        return ['f', 'y1_v1_y2_v2']     


    def network_output_spec(self):
        """
        Network output spec.

        Return:
            *inputs (list[str]): List of key(str) for the network outputs.
        """
        return ['f1_f2_f', 'log_std']


    def get_tensors(self):
        return dict(l_ts=self.l_ts, 
                    debug_ts=self.debug_ts)


    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['l_ts']
        del new_dict['debug_ts']
        return new_dict

