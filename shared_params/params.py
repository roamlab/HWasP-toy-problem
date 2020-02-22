import numpy as np

# env params
# in SI
half_force_range = 5.0
pos_range = 0.5
half_vel_range = 2.0

m1 = 0.1
m2 = 0.1
h = 0.2
l = 0.1
g = 9.8
dt = 0.002
n_steps_per_action = 5
n_steps_per_episode = 1000

n_springs = 1 # for multi-spring cases

reward_alpha = 10.0
reward_beta = 0.5
reward_gamma = 2.0
reward_switch_pos_vel_thresh = 0.1

# policy params
def inv_sigmoid(y, lb, ub):
    '''
    The inverse of the sigmoid scaling transform y = 1/(1+exp(-x)) * (ub - lb) + lb
    '''
    return -np.log((ub - lb) / (y - lb) - 1)

k_ub = 100.0 / n_springs
k_lb = 0.0 / n_springs
k_range = k_ub - k_lb
k_init = 50.0 / n_springs
k_pre_init = inv_sigmoid(k_init, k_lb, k_ub)
k_pre_init_lb = -5
k_pre_init_ub = 5

# init stds
std_range_ratio_action = 0.3
std_range_ratio_auxiliary = 0.3

f_std_init_action = std_range_ratio_action * (half_force_range * 2)
f_log_std_init_action = np.log(f_std_init_action)
k_std_init_action = std_range_ratio_action * k_range
k_log_std_init_action = np.log(k_std_init_action)

f_std_init_auxiliary = std_range_ratio_auxiliary * (half_force_range * 2)
f_log_std_init_auxiliary = np.log(f_std_init_auxiliary)
k_std_init_auxiliary = std_range_ratio_auxiliary * k_range
k_log_std_init_auxiliary = np.log(k_std_init_auxiliary)

# learning params
comp_policy_network_size = (32, 32)
# baseline_network_size = (32, 32)

ppo_algo_kwargs = dict(
    max_path_length=n_steps_per_episode,
    discount=0.99,
    gae_lambda=0.95,
    lr_clip_range=0.1,
    max_kl_step=0.01,

    optimizer_args=dict(
        batch_size=128,
        max_epochs=10,
        learning_rate=1e-3,
    ),
    stop_entropy_gradient=False,
    entropy_method='regularized',
    policy_ent_coeff=1e-4,
    center_adv=True,
)

ppo_train_kwargs = dict(n_epochs=1000, batch_size=2048, plot=False)