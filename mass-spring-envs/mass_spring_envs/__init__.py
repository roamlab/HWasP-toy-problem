from gym.envs.registration import register

register(
    id='MassSpringEnv_OptK_MultiSprings_HwAsPolicy-v1',
    entry_point='mass_spring_envs.envs:MassSpringEnv_OptK_MultiSprings_HwAsPolicy',
    max_episode_steps=1000,
)


register(
    id='MassSpringEnv_OptK_MultiSprings_HwAsAction-v1',
    entry_point='mass_spring_envs.envs:MassSpringEnv_OptK_MultiSprings_HwAsAction',
    max_episode_steps=1000,
)