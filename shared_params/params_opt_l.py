import numpy as np

# env params
# in SI
half_force_range = 5.0
pos_range = 0.5
half_vel_range = 2.0

m1 = 0.1
m2 = 0.1
h = 0.2
k = 20
g = 9.8
dt = 0.002
n_steps_per_action = 5
n_steps_per_episode = 1000

n_segments = 10 

reward_alpha = 1.0
reward_beta = 0.05
reward_gamma = 2.0
reward_switch_pos_vel_thresh = 0.08

# policy params
def inv_sigmoid(y, lb, ub):
    '''
    The inverse of the sigmoid scaling transform y = 1/(1+exp(-x)) * (ub - lb) + lb
    '''
    return -np.log((ub - lb) / (y - lb) - 1)

l_ub = 0.5 / n_segments
l_lb = 0.0 / n_segments
l_range = l_ub - l_lb
l_init = (l_lb + l_ub) / 2 # will be overrided
l_pre_init = inv_sigmoid(l_init, l_lb, l_ub)
l_pre_init_lb = -5
l_pre_init_ub = 5

k_interface = 2e2
b_interface = 1e1

# init stds
std_range_ratio_action = 1.0
std_range_ratio_auxiliary = 1.0

f_std_init_action = std_range_ratio_action * (half_force_range * 2)
f_log_std_init_action = np.log(f_std_init_action)
l_std_init_action = std_range_ratio_action * l_range
l_log_std_init_action = np.log(l_std_init_action)

f_std_init_auxiliary = std_range_ratio_auxiliary * (half_force_range * 2)
f_log_std_init_auxiliary = np.log(f_std_init_auxiliary)
l_std_init_auxiliary = std_range_ratio_auxiliary * l_range
l_log_std_init_auxiliary = np.log(l_std_init_auxiliary)

# learning params
comp_policy_network_size = (32, 32)
# baseline_network_size = (32, 32)

# for pure ppo
ppo_algo_kwargs = dict(
    max_path_length=n_steps_per_episode,
    discount=0.99,
    gae_lambda=0.95,
    lr_clip_range=0.1,

    optimizer_args=dict(
        batch_size=128,
        max_epochs=10,
        learning_rate=1e-3,
    ),
    stop_entropy_gradient=False,
    entropy_method='regularized',
    policy_ent_coeff=5e-4,
    center_adv=True,
)

ppo_train_kwargs = dict(n_epochs=2000, batch_size=2000, plot=False)

# for pure cmaes
cmaes_algo_kwargs = dict(
    max_path_length=n_steps_per_episode,
    n_samples=32,
    discount=0.99,
    sigma0=2.0,
)

cmaes_train_kwargs = dict(n_epochs=100, batch_size=2000, plot=False)


# for cmaes (hyperparam search) + ppo
ppo_inner_train_kwargs = dict(n_epochs=300, batch_size=500, plot=False)

ppo_inner_final_average_discounted_return_window_size = 10

cmaes_options = {'tolfun':1.0, 'tolx':0.001, 'popsize': 8, 'maxiter':5, 'verb_log': 1, 'bounds': [[l_lb,] * n_segments, [l_ub,] * n_segments]}
cmaes_x0 = [(l_lb + l_ub) / 2,] * n_segments
cmaes_sigma0 = (l_ub - l_lb) / 4  # init sigma ususally chosen as a quater of the total range
