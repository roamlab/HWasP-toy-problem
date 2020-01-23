# hw-sw-co-opt-toy-problem
The 1D mass-spring toy problem of hardware-software co-optimization using RL
```
        --------------
        ^      \
        |       \
        |       /  spring, stiffness k, masless, zero natural length 
        |      /
        |      \
        |       \
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
- Goal: optimize the computational policy f and the mechanical structure to drag m2 to the goal point with minimum effort,
i.e. reward is: 
```
-alpha*(y2-h)**2 - beta*f**2
``` 
- 3 optimization cases: optimize for
    - comp. policy and spring stiffness k
    - comp. policy and mass m1
    - comp. policy and bar length l

This repo includes:
- launchers: garage launchers
- mass-spring-envs: the gym environments for the mass-spring system with different optimization goals
- policies: the computational graphs and garage policies
- policy_players: replay the trained policy and plot the observations and actions