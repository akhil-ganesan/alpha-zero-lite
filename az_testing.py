import copy
from copy import deepcopy

import torch

from RLA.agent import simulate_games
from RLA.az_agent import AZAgent, PolicyValueNetwork, Config

if __name__ == "__main__":

    config = Config()
    games = 1

    agents = []
    for i in range(0, 1001, 200):
        pvnet = PolicyValueNetwork(config)
        checkpoint = torch.load(f'pvnet_{i}.tar')
        pvnet.load_state_dict(checkpoint['model_state_dict'])
        agents.append(AZAgent(pvnet, 100))

    res = [[None] * len(agents) for _ in agents]
    scores = copy.deepcopy(res)

    for i in range(len(res)):
        for j in range(len(res)):
            # print(i, j)
            w, l, t = simulate_games(agents[i], agents[j], games)
            res[i][j] = (w, l, t)
            scores[i][j] = (w - l) / games

    print([sum(i) for i in scores])
    print(scores)
    print(res)