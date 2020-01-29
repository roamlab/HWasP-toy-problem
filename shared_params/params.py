import numpy as np

# init values
def inv_sigmoid(y, lb, ub):
    '''
    The inverse of the sigmoid scaling transform y = 1/(1+exp(-x)) * (ub - lb) + lb
    '''
    return -np.log((ub - lb)/ (y - lb) - 1)

k_ub = 100.0
k_lb = 0.0
k_range = k_ub - k_lb
k_init = 50
k_pre_init = inv_sigmoid(k_init, k_lb, k_ub)

std_init = 5.0
log_std_init = np.log(std_init)




# training params
comp_policy_network_size = (32, 32)
baseline_network_size = (32, 32)

ppo_algo_kwargs = dict(
    max_path_length=500,
    discount=0.99,
    gae_lambda=0.95,
    lr_clip_range=0.1,
    optimizer_args=dict(
        batch_size=64,
        max_epochs=10,
    ),

    # stop_entropy_gradient=True,
    # entropy_method='max',
    # policy_ent_coeff=0.03,
    # center_adv=False,

    stop_entropy_gradient=False,
    entropy_method='regularized',
    policy_ent_coeff=1e-3,  # working ok with 2e-3 but explode at some point
    center_adv=True,
)

ppo_train_kwargs = dict(n_epochs=500, batch_size=2048, plot=False)



# env params
# in SI
force_range = 10.0
pos_range = 1.0
vel_range = 1.0

m1 = 0.2
m2 = 0.2
h = 0.2
l = 0.1
g = 9.8
dt = 0.001
n_steps_per_action = 10

reward_alpha = 10.0
reward_beta = 0.01

