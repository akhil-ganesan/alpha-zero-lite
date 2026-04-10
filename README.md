# Alpha Zero

This repo contains a recreation of the Alpha Zero reinforcement learning system. It was trained and tested on a CPU to play tic tac toe, although the algorithm is interoperable to any game/problem if compute resources are avaliable.

## Background

### RL Environment

Mechanics for tic tac toe and agent play are located in [env.py](env.py). For the tic tac toe game mechanics, each player has a 3x3 matrix storing their occupied positions and a list containing how many squares of the 8 possible winning configurations (3 across, 3 down, 2 diagonal) are filled, allowing for slightly quicker game ending determination. Within the RL environment wrapper, the game is initialized and agents are called in alteration until the game terminates.

### Baseline Agents

Prior to the Alpha Zero agent, 3 other agents were developed in [agent.py](agent.py):
* **RandomAgent**: Selects random moves
* **Minimax Agent**: Applies alpha-beta pruning in a minimax tree selection (without a depth limit due to the small size of tic tac toe)
  * An additional parameter "rmt" (random move threshold) is included; while there are more moves avaliable than rmt, random moves are made. This allows for random first moves and potential exploits by other agents earlier in the game
* **Monte Carlo Tree Search (MCTS) Agent**: Applies the monte carlo tree search (with a random-action rollout policy) using the upper confidence bound (UCB) heuristic

## Alpha Zero Structure

The Alpha Zero system incorporates a neural network with MCTS. This code is located in [az_agent.py](az_agent.py). Essentially, the neural network is used to estimate the optimal policy and value of the states in the game (modeled as an markov decision process); MCTS is used to generate data for training the network as a form of policy-value iteration.

### Policy-Value Network

The neural network consists of a residual convolutional network body with 2 heads for policy prediction & value estimation. The unbatched input to the network is a 2x3x3 tensor (2 planes of a binary 3x3 matrix representing board placements for each player; note the player actively playing is in the first plane). The policy output is a tensor of size 9 representing the probabilities of placing in each position on the board. The value output is a size 1 tensor estimating the game outcome from the active position for the player playing (between -1 to 1).

### MCTS

The MCTS algorithm is similar to the baseline MCTS implementation with the following exceptions:
* When each node is expanded, if the node isn't terminal, the policy-value network is called to estimate the node's value & optimal policy at the node (i.e. no rollouts are ran)
  * Dirichlet noise is added to the output policy to encourage exploration ($\alpha=\frac{10}{n}$, where $n$ is the average number of possible moves in each position, which for tic tac toe can be estimated as $n \approx \frac{9}{2}$)
  * The policy output is masked to exclude illegal moves based on the game dynamics (this is a key distinction between the AlphaZero & MuZero algorithm)
* Tree searches/traversals were based on 2 different heuristic:
  * The pUCT heuristic $h = Q + c_{pUCT} * P * \frac{\sqrt{N_t}}{1 + N}$, where for a node, $Q$ is the state's utility (updated during MCTS backpropogation), $P$ is the prior probability (i.e. for the node's parent, it's the probability associated with the transition to the specific child state/node from policy output from the policy-value network at the parent node), $N_t$ is the total number of tree visits to each child node of the current node's parent (also equal to the number of visits to the parent node - 1), $N$ is the number of visits to the node, & $c_{pUCT}$ is a constant balancing exploration of new moves to exploitation of explored moves to maximize utility. This heuristic was used in the Alpha Go Zero implementation
  * An alternative heuristic tested was $h = \bar{Q} + P * \frac{\sqrt{N_t}}{1 + N} * (c_1 + \log{\frac{N_t + c_2 + 1}{c_2}})$. This has 2 main differences from the first heuristic:
    * $\bar{Q}$ is used instead of $Q$; this is the normalized utility calculated by scaling $Q$ to be between 0 and 1 based on $Q_{max} = 1$ and $Q_{min} = -1$ (the bounding utilities can also be tracked as the extrema throughout the tree)
    * $c_{pUCT}$ is replaced with the expression $c_1 + \log{\frac{N_t + c_2 + 1}{c_2}}$. This essentially represents a $c_{pUCT}$ initially set to $c_1$ that increases throughout the tree search based on $c_2$ (allowing for greater valuing of unexplored states later in the tree search promoting more balanced search)
  * Note the state's utility stored in each node is relative to the player playing; as a result, to adequately reflect utility maximization for the parent node in the heuristics, the values of the children nodes are negated to represent the state value relative to the parent node (the player that played prior); this is possible because the game is a 2-player & 0-sum

## Training

The policy-value network was trained through iterative self-play: a game would be played out between 2 agents with the same policy-value network. The states of each position are stored along with the normalized visit counts to each child of MCTS root node (representing probabilities for each action at that state i.e. the optimal policy) & the final game outcome for the player playing (essentially a vector of 0s for a tie or alternating between -1 and 1 otherwise, with the start value based on if player 1 won). The network is then trained by feeding in the batch of the single game's states, calculating loss between the output values (using mean squared error) & policies (using cross entropy), & backpropogating.

Training was done using an AdamW optimizer with the learning rate set to 0.001 for 4,000 "epochs" (i.e. games) on a cpu, taking around an hour. Additionally, during training, instead of MCTS returning the most-visited action deterministically, the action returned is sampled by a probability distribution of the final roots childrens' visit counts. Furthermore, this policy is exponentiated and renormalized by $\frac{1}{T}$, where $T$ is the temperature, which is reduces from 1 to 0.25 throughout the training process linearly. Note: the higher the temperature is, the more uniform the distribution is, promoting more exploration earlier in the training process. The temperature logic is implemented within the agent, but the training loop is implemented within [az_training.py](az_training.py).


## Results

Models were grouped into 5 generations, with each generation separated by 1,000 games of training from self-play. After training, the models were cross-compared by playing games against each other; all models drew against each other when using the UCB heuristic (indicating there was no regression in model performance) and all models after generation 0 tied and beat generation 0 when using the pUCT heuristic (indicating improvement across generations). Each generation was also compared against the 3 baseline models (playing games as first and second player); the results from this generally support the conclusion that training converged after the first generation (where the greatest score improvements came). These results can be seen under the [UCB Testing (c2=0.625)](https://github.com/akhil-ganesan/alpha-zero-3t/tree/main/UCB%20Testing%20(c2%3D0.625)) & [pUCT Testing](https://github.com/akhil-ganesan/alpha-zero-3t/tree/main/pUCT%20Testing) folders.


# Next Steps

* Add high-performance computing support for training (e.g. CUDA/GPU support, parallelization, etc.)
* Implement asynchronous, parallel game playing & model training
  * Store data in a separate replay buffer
  * Augment data (because tic tac toe is rotationally & reflectively symmetric, each position can be rotated 90 degrees & reflected across the x & y axis to generate more positions with equivalent valuations to train the network)
* Train/Test on a more difficult, unsolved game (parameterizing the game played within the RL environment)
* Implement the MuZero algorithm based on another neural network trained to master game dynamics

# References

1. [AlphaGo Paper](www.nature.com/articles/nature16961)
2. [AlphaZero Paper](https://arxiv.org/pdf/1712.01815)
3. [AlphaGo Zero Paper](https://gwern.net/doc/reinforcement-learning/model/alphago/2017-silver.pdf)
4. [Mu Zero Paper](https://arxiv.org/pdf/1911.08265)
5. [ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html)
6. Example Implementations: [AlphaZero](https://github.com/CogitoNTNU/AlphaZero), [MuZero 1](https://github.com/werner-duvaud/muzero-general), [MuZero 2](https://github.com/michaelnny/muzero)
7. [Article](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5) on Dirichlet Noise in Small Games
