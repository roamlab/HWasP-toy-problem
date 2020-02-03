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

from mass_spring_envs.envs.mass_spring_env_opt_k_hw_as_policy import MassSpringEnv_OptK_HwAsPolicy
from policies.opt_k_hw_as_policy.mech_policy_model import MechPolicyModel
from policies.opt_k_hw_as_policy.comp_mech_policy import CompMechPolicy_OptK_HwAsPolicy

from shared_params import params


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(MassSpringEnv_OptK_HwAsPolicy())

        comp_policy_model = MLPModel(output_dim=1, 
            hidden_sizes=params.comp_policy_network_size, 
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.tanh)

        mech_policy_model = MechPolicyModel(k_pre_init=params.k_pre_init, log_std_init=[params.f_log_std_init, params.f_log_std_init])

        policy = CompMechPolicy_OptK_HwAsPolicy(name='comp_mech_policy', 
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

        runner.train(**params.ppo_train_kwargs)

    
if __name__=='__main__':
    run_experiment(run_task, snapshot_mode='last', seed=0, force_cpu=True)
