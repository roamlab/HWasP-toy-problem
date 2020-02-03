import numpy as np

# env params
# in SI
half_force_range = 10.0
pos_range = 1.0
half_vel_range = 1.0

m1 = 0.2
m2 = 0.2
h = 0.2
l = 0.1
g = 9.8
dt = 0.001
n_steps_per_action = 10

reward_alpha = 10.0
reward_beta = 0.01


# init values
def inv_sigmoid(y, lb, ub):
    '''
    The inverse of the sigmoid scaling transform y = 1/(1+exp(-x)) * (ub - lb) + lb
    '''
    return -np.log((ub - lb)/ (y - lb) - 1)

k_ub = 100.0
k_lb = 0.0
k_range = k_ub - k_lb
k_init = 10.0
k_pre_init = inv_sigmoid(k_init, k_lb, k_ub)


# init stds
std_range_ratio = 0.2

f_std_init = std_range_ratio * (half_force_range * 2)
f_log_std_init = np.log(f_std_init)
k_std_init = std_range_ratio * k_range
k_log_std_init = np.log(k_std_init)



# training params
comp_policy_network_size = (32, 32)
# baseline_network_size = (32, 32)

ppo_algo_kwargs = dict(
    max_path_length=500,
    discount=0.99,
    gae_lambda=0.95,
    lr_clip_range=0.1,
    max_kl_step=0.01,

    optimizer_args=dict(
        batch_size=64,
        max_epochs=10,
        learning_rate=1e-3,
    ),
    stop_entropy_gradient=False,
    entropy_method='regularized',
    policy_ent_coeff=1e-5,
    center_adv=True,
)

ppo_train_kwargs = dict(n_epochs=500, batch_size=2048, plot=False)