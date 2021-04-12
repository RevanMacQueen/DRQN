from utils.visualizations import *
from utils.data_extractors import *

root = Path('results_batch_size')
params, episode_len = extract_epsisode_lengths(root, ['model_arch', 'env', 'seed', 'learning_freq', 'target_update_freq', 'seq_len'])


env_rename = {
    'CartPole-v1': 'CartPole',
    'envs:random_maze-v0' : 'Maze',
    'MountainCar-v0' : 'MountainCar'
                }

# BAR PLOTS
for env in ['CartPole-v1']:
    inds = np.where((params[:, 1]==env) ) # right env and seq_len 
    params_ = params[inds]
    episode_len_ = episode_len[inds]
    plot_avg_episode_length(params_, episode_len_, categories=[3, 4], shapes=[0,5], tick_fmt='%s, %s', shape_fmt = '%s, %s', xlabel='Learning frequency, target update frequency', ylabel='Average episode length', title=env_rename[env], scale='log2')
    plt.savefig("figures/batch_size_%s_same_layer.pdf" % env_rename[env], bbox_inches='tight')
    plt.clf()

# mountain car
# env = 'MountainCar-v0'
# inds = np.where((params[:, 1]==env) ) # right env and seq_len 
# params_ = params[inds]
# episode_len_ = episode_len[inds]
# plot_avg_episode_length(params_, episode_len_, categories=[3, 4], shapes=[0,5], tick_fmt='%s, %s', shape_fmt = '%s, %s', xlabel='Learning frequency, target update frequency', ylabel='Average episode length', title=env_rename[env])
# plt.savefig("figures/episode_len_%s_same_layer.pdf" % env_rename[env], bbox_inches='tight')
# plt.clf()

# root = Path('results_eplen')

# # LEARNING CURVES
# for env in ['CartPole-v1', 'MountainCar-v0']:
#     filters = {
#     'env': [env],
#     'learning_freq' : [1],
#     'target_update_freq' : [100]}

#     params, times = extract_epsisode_lengths(root, ['model_arch', 'env', 'seed', 'learning_freq', 'target_update_freq', 'seq_len'], filters)

#     plot_rewards(params, times, plot=[0,5], plot_fmt='%s, %s', xlabel="Episode", ylabel="Episode length", title=env_rename[env])
#     plt.savefig("figures/learning_curves/%s_1_100_rolling.pdf" % env_rename[env], bbox_inches='tight')
#     plt.clf()

# root = Path('results_eplen_maze')
# env ='envs:random_maze-v0'
# filters = {
# 'env': [env],
# 'learning_freq' : [1],
# 'target_update_freq' : [100]}

# params, rewards = extract_epsisode_lengths(root, ['model_arch', 'env', 'seed', 'learning_freq', 'target_update_freq', 'seq_len'], filters)
# plot_rewards(params, rewards, plot=[0,5], plot_fmt='%s, %s', xlabel="Episode", ylabel="Episode length", title=env_rename[env])
# plt.savefig("figures/learning_curves/%s_1_100_rolling.pdf" % env_rename[env], bbox_inches='tight')
# plt.clf()
