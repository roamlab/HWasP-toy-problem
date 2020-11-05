import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # only show warning and errors in TF
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # only use CPU
import tensorflow as tf
import numpy as np
import pathlib
import dateutil
import dowel
from dowel import logger, tabular
from datetime import datetime
import argparse

from garage.tf.envs import TfEnv
from garage.tf.models.mlp_model import MLPModel

from my_garage.algos.ars import ARS

from mass_spring_envs.envs.mass_spring_env_opt_l import MassSpringEnv_OptL_HwAsAction
from policies.opt_l.models import MechPolicyModel_OptL_HwAsAction
from policies.opt_l.policies import CompMechPolicy_OptL_HwAsAction

# from launchers.utils.zip_project import zip_project

from shared_params import params_opt_l as params


class DowelManager:
    """
        This is kinda wierd since this is actually handling global
        resource. This is just a context manager to avoid mannually
        catching the exception for handling dowel.
    """

    def __init__(self, exp_prefix='exp', log_dir='./data/local'):
        log_dir = os.path.join(log_dir, exp_prefix)

        now = datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
        exp_name = '{}_{}'.format(exp_prefix, timestamp)

        log_dir = os.path.join(log_dir, exp_name)

        self.log_dir = log_dir
        self.exp_name = exp_name
        self.model_path = os.path.join(self.log_dir, 'models')
        pathlib.Path(self.model_path).mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        tabular_log_file = os.path.join(self.log_dir, 'progress.csv')
        text_log_file = os.path.join(self.log_dir, 'debug.log')

        logger.add_output(dowel.TextOutput(text_log_file))
        logger.add_output(dowel.CsvOutput(tabular_log_file))
        logger.add_output(dowel.TensorBoardOutput(self.log_dir))
        logger.add_output(dowel.StdOutput())

        logger.push_prefix('[%s] ' % self.exp_name)
        return self

    def __exit__(self, type, value, traceback):
        logger.remove_all()
        logger.pop_prefix()


def run_ars(exp_prefix, seed):
    env = TfEnv(MassSpringEnv_OptL_HwAsAction(params))

    with tf.compat.v1.Session() as sess:
        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=params.comp_policy_network_size, 
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.tanh,
            )
        mech_policy_model = MechPolicyModel_OptL_HwAsAction(params)
        policy = CompMechPolicy_OptL_HwAsAction(name='comp_mech_policy', 
                env_spec=env.spec, 
                comp_policy_model=comp_policy_model, 
                mech_policy_model=mech_policy_model)

        ars = ARS(env_name=None,
                env=env,
                policy_params=None,
                policy=policy,
                seed = seed,
                **params.ars_kwargs)
        
        with DowelManager(exp_prefix=exp_prefix) as manager:    
            ars.train(params.ars_n_iter, dump=True)


if __name__ == '__main__':

    now = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=int(now.timestamp()), type=int, help='seed')
    parser.add_argument('--exp_id', default=now.strftime("%Y_%m_%d_%H_%M_%S"), help='experiment id (suffix to data directory name)')

    args = parser.parse_args()

    run_ars(exp_prefix='ars_opt_l_hw_as_action_{}_'.format(args.exp_id) + str(params.n_segments)+'_params', seed = args.seed)
