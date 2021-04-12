from gym.envs.registration import register

register(
    id='random_maze-v0',
    entry_point='envs.random_maze:RandomMaze',
)

register(
    id='MountainCar1000-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=1000,
    reward_threshold=-110.0,
)