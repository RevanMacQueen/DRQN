from utils.visualizations import *
from utils.data_extractors import *


root = Path('results')
params, episode_len = extract_epsisode_lengths(root, ['model_arch', 'env', 'seed', 'learning_freq', 'target_update_freq'])

env_rename = {
    'CartPole-v1': 'cartpole',
    'envs:random_maze-v0' : 'maze',
    'MountainCar-v0' : 'mountaincar'
                }

for model_arch in ['RNN', 'FFN']:
    for env in ['envs:random_maze-v0', 'CartPole-v1', 'MountainCar-v0']:
        inds = np.where((params[:, 0]==model_arch) & (params[:, 1]==env))
        print(inds)
        params_, episode_len_ = params[inds], episode_len[inds]
        print(params_.shape)

        plot_episode_len(params_, episode_len_, 3, 4, 2, '', '')
        plt.savefig("figures/episode_len_%s_%s.pdf" %(model_arch, env_rename[env]), bbox_inches='tight')

        plt.clf()
