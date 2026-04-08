import copy

import torch

from agent import simulate_games, RandomAgent, MiniMaxAgent, MCTSAgent
from az_agent import PolicyValueNetwork, Config
from az_agent import AZAgent

if __name__ == "__main__":

    config = Config()
    games = 2

    agents = []
    for i in [0, 1000, 2000, 3000, 4000]: # range(0, 1001, 200): # 4800, 6800
        pvnet = PolicyValueNetwork(config)
        checkpoint = torch.load(f'pvnet_v4_{i}.tar')
        pvnet.load_state_dict(checkpoint['model_state_dict'])
        agents.append(AZAgent(pvnet, 100))

    res = [[None] * len(agents) for _ in agents]
    scores = copy.deepcopy(res)

    for i in range(len(res)):
        for j in range(len(res)):
            if i != j:
                w, l, t = simulate_games(agents[i], agents[j], games, output=False)
                res[i][j] = (w, l, t)
                scores[i][j] = (w - l) / games
            else:
                scores[i][j] = 0

    print("Cross-Evaluation Results")
    for i in range(len(scores)):
        print(f"Gen {i}: {sum(scores[i])} \t {scores[i]}")

    print("Baseline Agent Results")
    games = 10
    for i, agent in enumerate(agents):
        # Testing against random
        w1, l1, _ = simulate_games(agent, RandomAgent(), games, output=False)
        w2, l2, _ = simulate_games(RandomAgent(), agent, games, output=False)
        random_score = w1+l2-l1-w2

        # Testing against exploitable minimax
        rmt = 7 # 5
        w1, l1, _ = simulate_games(agent, MiniMaxAgent(rmt=rmt), games, output=False)
        w2, l2, _ = simulate_games(MiniMaxAgent(rmt=rmt), agent, games, output=False)
        minimax_score = w1 + l2 - l1 - w2

        # Testing against MCTS
        w1, l1, _ = simulate_games(agent, MCTSAgent(), games, output=False)
        w2, l2, _ = simulate_games(MCTSAgent(), agent, games, output=False)
        mcts_score = w1 + l2 - l1 - w2

        print(f"Gen {i}: {(random_score + minimax_score + mcts_score, random_score, minimax_score, mcts_score)}")
