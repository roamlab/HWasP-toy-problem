#!/usr/bin/env python3

import gym

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # only show warning and errors in TF
import tensorflow as tf
import numpy as np

from garage.envs import normalize
from garage.experiment import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.np.baselines import LinearFeatureBaseline

from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.models import MLPModel

import mass_spring_envs
from policies.opt_spring_stiffness.mech_policy_model import MechPolicyModel
from policies.opt_spring_stiffness.policy import CompMechPolicy_OptSpringStiffness


def inv_sigmoid(y, lb, ub):
    '''
    The inverse of the sigmoid scaling transform y = 1/(1+exp(-x)) * (ub - lb) + lb
    '''
    return -np.log((ub - lb)/ (y - lb) - 1)

k_init = 5e1
ub = 1e2
lb = 0.0
k_pre_init = inv_sigmoid(k_init, lb, ub)

std_init = 1.0
log_std_init = np.log(std_init)

def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(normalize(gym.make('mass_spring_envs:MassSpringEnv_OptSpringStiffness-v1')))


        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=(32, 32), 
            hidden_nonlinearity=tf.nn.tanh)
        mech_policy_model = MechPolicyModel(k_pre_init=k_pre_init, log_std_init=log_std_init)

        policy = CompMechPolicy_OptSpringStiffness(name='test_comp_mech_policy', 
                env_spec=env.spec, 
                comp_policy_model=comp_policy_model, 
                mech_policy_model=mech_policy_model)

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(32, 32),
                use_trust_region=True,
            ),
        )
        # baseline = LinearFeatureBaseline(env_spec=env.spec)

        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=128,
                max_epochs=10,
            ),
            # stop_entropy_gradient=True,
            # entropy_method='max',
            # policy_ent_coeff=0.02,
            # center_adv=False,

            stop_entropy_gradient=False,
            entropy_method='regularized',
            policy_ent_coeff=1e-3,  # working ok with 2e-3 but explode at some point
            center_adv=True,

        )

        runner.setup(algo, env)

        runner.train(n_epochs=100, batch_size=4096, plot=False)

    
if __name__=='__main__':
    run_experiment(run_task, snapshot_mode='last', seed=1, n_parallel=1, force_cpu=True)