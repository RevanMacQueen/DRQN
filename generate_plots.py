from utils.visualizations import *
from utils.data_extractors import *


root = Path('results')
params, episode_len = extract_epsisode_lengths(root, ['model_arch', 'env', 'seed'])
plot_episode_len(params, episode_len, 0, 1, 2, 'Episode', 'Num Timesteps')
plt.savefig("episode_len.pdf", bbox_inches='tight')