from gym.envs.registration import register

register(
    id='MassSpringEnv_OptSpringStiffness-v1',
    entry_point='mass_spring_envs.envs:MassSpringEnv_OptSpringStiffness',
)