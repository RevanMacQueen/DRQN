from gym.envs.registration import register

register(
    id='random_maze-v0',
    entry_point='envs.random_maze:RandomMaze',
)
