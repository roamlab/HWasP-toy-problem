import gym

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # only show warning and errors in TF
import tensorflow as tf
import numpy as np
import pandas as pd

from garage.envs import normalize
from garage.experiment import run_experiment
from garage.tf.algos.ppo import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.np.baselines import LinearFeatureBaseline

from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.models.mlp_model import MLPModel

import cma

from mass_spring_envs.envs.mass_spring_env_opt_l import MassSpringEnv_OptL_HwAsAction
from policies.opt_l.models import MechPolicyModel_OptL_FixedHW
from policies.opt_l.policies import CompMechPolicy_OptL_HwAsAction

from shared_params import params_opt_l as params

from launchers.utils.zip_project import zip_project
from launchers.utils.normalized_env import normalize

from datetime import datetime
import sys
import argparse

global l_pre_init

def run_task(snapshot_config, *_):
    """Run task."""
    global l_pre_init
    params.l_pre_init = l_pre_init

    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        # env = TfEnv(normalize(MassSpringEnv_OptL_HwAsAction(params), normalize_action=False, normalize_obs=False, normalize_reward=True, reward_alpha=0.1))
        env = TfEnv(MassSpringEnv_OptL_HwAsAction(params))

        # zip_project(log_dir=runner._snapshotter._snapshot_dir)

        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=params.comp_policy_network_size, 
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.tanh,
            )
        
        mech_policy_model = MechPolicyModel_OptL_FixedHW(params)

        policy = CompMechPolicy_OptL_HwAsAction( # reused policy of HWasAction
            name='comp_mech_policy', 
            env_spec=env.spec, 
            comp_policy_model=comp_policy_model, 
            mech_policy_model=mech_policy_model)

        # baseline = GaussianMLPBaseline(
        #     env_spec=env.spec,
        #     regressor_args=dict(
        #         hidden_sizes=params.baseline_network_size,
        #         hidden_nonlinearity=tf.nn.tanh,
        #         use_trust_region=True,
        #     ),
        # )
        
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        
        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            **params.ppo_algo_kwargs
        )

        runner.setup(algo, env)

        runner.train(**params.ppo_inner_train_kwargs)

    tf.compat.v1.reset_default_graph()



def cmaes_obj_fcn(l_init, exp_prefix):
    global l_pre_init
    l_pre_init = params.inv_sigmoid(l_init, params.l_lb, params.l_ub)

    now = datetime.now()
    exp_name = now.strftime("%Y_%m_%d_%H_%M_%S")

    run_experiment(run_task, exp_prefix=exp_prefix, exp_name=exp_name, snapshot_mode='last', seed=args.seed, force_cpu=True)

    csv_path = os.path.join(os.environ['PROJECTDIR'], 'data/local', exp_prefix.replace('_', '-'), exp_name, 'progress.csv')
    csv_df = pd.read_csv(csv_path)
    final_avg_discounted_return = np.mean(csv_df['AverageDiscountedReturn'][-params.ppo_inner_final_average_discounted_return_window_size:])
    return -final_avg_discounted_return



if __name__=='__main__':

    now = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=int(now.timestamp()), type=int, help='seed')
    parser.add_argument('--exp_id', default=now.strftime("%Y_%m_%d_%H_%M_%S"), help='experiment id (suffix to data directory name)')

    args = parser.parse_args()

    exp_prefix='cmaes_ppo_opt_l_{0}_{1}_params/seed_{2}'.format(args.exp_id, params.n_segments, args.seed)

    # CMA-ES global optimization
    options = params.cmaes_options
    options['seed'] = args.seed
    options['verb_filenameprefix'] = os.path.join(os.environ['PROJECTDIR'], 'data/local', exp_prefix.replace('_', '-'), '-')
    x0 = params.cmaes_x0
    sigma0 = params.cmaes_sigma0

    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    es.optimize(cmaes_obj_fcn, args=[exp_prefix])
    es.result_pretty()

    zip_project(log_dir=os.path.join(os.environ['PROJECTDIR'], 'data/local', exp_prefix.replace('_', '-')))

