# Alpha Zero

This repo contains a recreation of the Alpha Zero reinforcement learning system. It was trained and tested on a CPU to play tic tac toe, although the algorithm is interoperable to any game/problem if compute resources are avaliable.

## Background

### RL Environment



### Non-AI Agents

In addition to the alpha zero agent, 3 other agents were developed in [agent.py](agent.py):
* **RandomAgent**: Selects random moves
* **Minimax Agent**: Applies alpha-beta pruning in a minimax tree selection (without a depth limit due to the small size of tic tac toe)
  * An additional parameter "rmt" (random move threshold) is included; while there are more moves avaliable than rmt, random moves are made. This allows for random first moves and potential exploits earlier in the game for the perfect agent
* **Monte Carlo Tree Search (MCTS) Agent**: Applies the monte carlo tree search (parameterizing rollouts and simulations) using the upper confidence bound (UCB) heuristic

## Alpha Zero Structure

* 

## Training

* 

## Results



# Next Steps

* Add high-performance computing support for training (e.g. CUDA/GPU support, parallelization, etc.)
* Implement asynchronous, parallel game playing & model training
  * Store data in a separate replay buffer
  * Augment data (because tic tac toe is rotationally & reflectively symmetric, each position can be rotated 90 degrees & reflected across the x & y axis to generate more positions with equivalent valuations to train the network)
* Train/Test on a more difficult, unsolved game

# References

1. [AlphaGo Paper](www.nature.com/articles/nature16961)
2. [AlphaZero Paper](https://arxiv.org/pdf/1712.01815)
3. [AlphaGo Zero Paper](https://gwern.net/doc/reinforcement-learning/model/alphago/2017-silver.pdf)
4. [Mu Zero Paper](https://arxiv.org/pdf/1911.08265)
5. [ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html)
6. Example Implementations: [AlphaZero](https://github.com/CogitoNTNU/AlphaZero), [MuZero 1](https://github.com/werner-duvaud/muzero-general), [MuZero 2](https://github.com/michaelnny/muzero)
7. [Article](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5) on Dirichlet Noise in Small Games
