# DRQN

Implementation of [DRQN](https://arxiv.org/abs/1507.06527) and [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) on fully observable problems. The goal of this project is to determine whether recurrent neural networks can always be used in place of feedforward networks for reinforcement learning, even when the environment is fully observable (meaning there is no need for recurrence). 

## Installation

First clone this repo:
```
git clone https://github.com/RevanMacQueen/DRQN.git
```

Then use pip to install dependencies. We recommend Python >= 3.8
```
pip3 install requirements.txt
```

Lastly install this package:
```
pip3 install -e .
```

## Usage

To run the code in this repo, run main.py. For example, to run DRQN on a randomly generated maze environment with default hyperparameters:
```
python3 main.py --env envs:random_maze-v0 --model_arch RNN '
```
This command will output results to a newly created directory results/.


