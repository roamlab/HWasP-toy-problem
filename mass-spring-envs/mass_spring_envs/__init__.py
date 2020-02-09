from gym.envs.registration import register

register(
    id='MassSpringEnv_OptK_HwAsPolicy-v1',
    entry_point='mass_spring_envs.envs:MassSpringEnv_OptK_HwAsPolicy',
    max_episode_steps=2000,
)


register(
    id='MassSpringEnv_OptK_HwAsAction-v1',
    entry_point='mass_spring_envs.envs:MassSpringEnv_OptK_HwAsAction',
    max_episode_steps=2000,
)