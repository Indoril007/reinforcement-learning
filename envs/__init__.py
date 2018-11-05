from gym.envs.registration import register

register(
    id='SimpleGridWorld-v0',
    entry_point='envs.gridworlds:SimpleGridWorld',
)