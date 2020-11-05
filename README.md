# HWasP-toy-problem
The 1D mass-spring toy problem of hardware-software co-optimization using RL. Presented in the paper: [Hardware as Policy: Mechanical and Computational Co-Optimization using Deep Reinforcement Learning](https://arxiv.org/abs/2008.04460)
```
        --------------
        ^    \    \
        |     \    \
        |     / .. /  springs, stiffness k1, k2 ..., masless, zero natural length 
        |    /    /
        |    \    \
        |     \    \
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
        |               |                   |
        |               |                   |
        |               |                   |
        v      Goal     v input f (or i)    v g
        --------------
```
## Problem statement:
- Input to the system: a force f (proportional to motor current i) on m2
- Goal: optimize the computational policy and the mechanical structure to:
    - Primary goal: drag m2 to the goal point and stay there 
    - Secondary goal: use minimum effort,
i.e. reward is: 
```
pos_penalty = alpha*abs(y2-h)
vel_penalty = beta*abs(v2)
force_penalty = gamma*abs(f)

if pos_penalty + vel_penalty > threshold: # meaning the primary goal is not achieved yet
    reward = -pos_penalty - vel_penalty - a_large_force_penalty
else: # meaning the primary goal is achieved
    reward = -pos_penalty - vel_penalty - force_penalty
``` 
- 2 optimization cases: optimize for
    - comp. policy and spring stiffness k (shown in the ASCII drawing above)
    - comp. policy and bar length l (shown in the ASCII drawing in the code)

## This repo includes:
- launchers: garage launchers for training and replay
- mass-spring-envs: the gym environments for the mass-spring system with different optimization goals
- my_garage: the implementation of Augmented Random Search (ARS) (adapted from the original paper) fitted to the garage framework
- policies: the computational graphs and garage policies
- scripts: the bash scripts to start launchers in different terminals in parallel
- shared_params: the files containing all parameters for this project (a centralized way of parameter management)

## Installation:
You need:
- tensorflow (1.15)
- numpy
- garage (2019.10.03)
- gym
- mass-spring-envs

You can isntall all you need at once by:
```
sh install.sh
```


## To launch the training, for example:
```
sh activate_env.sh
python lauchers/train/ppo_opt_k_hw_as_policy.py
```

(Please note: the terminology "HW as action" in the code is the same as the "HWasP-Minimal" presented in the paper)
