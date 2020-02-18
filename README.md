# hw-sw-co-opt-toy-problem
The 1D mass-spring toy problem of hardware-software co-optimization using RL
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
        |               |               |
        |               |               |
        |               |               |
        v      Goal     v input f       v g
        --------------
```
Problem statement:
- Input to the system: a force f on m2
- Goal: optimize the computational policy f and the mechanical structure to:
    - Primary goal: drag m2 to the goal point and stay there 
    - Secondary goal: with minimum effort,
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
- 3 optimization cases: optimize for
    - comp. policy and spring stiffness k
    - comp. policy and mass m1
    - comp. policy and bar length l

This repo includes:
- launchers: garage launchers for training and replay
- mass-spring-envs: the gym environments for the mass-spring system with different optimization goals
- policies: the computational graphs and garage policies
- scripts: the bash scripts to start launchers in different terminals in parallel
- shared_params: the files containing all parameters for this project (a centralized way of parameter management)

To run, you need:
- tensorflow
- numpy
- garage
- gym
- install the mass-spring-envs