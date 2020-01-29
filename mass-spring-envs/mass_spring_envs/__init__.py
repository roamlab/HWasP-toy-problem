from gym.envs.registration import register

register(
    id='MassSpringEnv_OptK_HwAsPolicy-v1',
    entry_point='mass_spring_envs.envs:MassSpringEnv_OptK_HwAsPolicy',
    max_episode_steps=500,
)