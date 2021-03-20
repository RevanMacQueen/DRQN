from utils.visualizations import *
from utils.data_extractors import *


# root = Path('results_tau_1')
# params, episode_len = extract_epsisode_lengths(root, ['model_arch', 'env', 'seed', 'learning_freq', 'target_update_freq'])

# env_rename = {
#     'CartPole-v1': 'cartpole',
#     'envs:random_maze-v0' : 'maze',
#     'MountainCar-v0' : 'mountaincar'
#                 }

# for model_arch in ['RNN', 'FFN']:
#     for env in ['envs:random_maze-v0', 'CartPole-v1', 'MountainCar-v0']:
#         inds = np.where((params[:, 0]==model_arch) & (params[:, 1]==env))
#         print(inds)
#         params_, episode_len_ = params[inds], episode_len[inds]
#         print(params_.shape)

#         plot_episode_len(params_, episode_len_, 3, 4, 2, '', '')
#         plt.savefig("figures/episode_len_%s_%s_tau_1.pdf" %(model_arch, env_rename[env]), bbox_inches='tight')

#         plt.clf()



root = Path('results_cartpole_200k_seq_len1')
params, episode_len = extract_epsisode_lengths(root, ['model_arch', 'env', 'seed', 'learning_freq', 'target_update_freq'])


env_rename = {
    'CartPole-v1': 'cartpole',
    'envs:random_maze-v0' : 'maze',
    'MountainCar-v0' : 'mountaincar'
                }
for env in ['CartPole-v1']: #['envs:random_maze-v0', 'CartPole-v1', 'MountainCar-v0']:
    inds = np.where( params[:, 1]==env)
    params_ = params[inds]
    episode_len_ = episode_len[inds]

    plot_avg_episode_length(params_, episode_len_, categories=[3, 4], shapes=0, tick_fmt='%s. %s', xlabel='learning freq., target update freq', ylabel='avg. episode length', title=env_rename[env])

    plt.savefig("figures/episode_len_%s_tau_1_seq_1.pdf" % env_rename[env], bbox_inches='tight')
    plt.clf()