from gym.envs.registration import register

register(
    id='random_maze-v0',
    entry_point='env.random_maze:RandomMaze',
)
