import copy

import torch

from RLA.agent import simulate_games
from RLA.az_agent import PolicyValueNetwork, Config
from RLA.cleaned_az_agent import AZAgent2

if __name__ == "__main__":

    config = Config()
    games = 2

    agents = []
    for i in [0, 5800, 7800]: # range(0, 1001, 200): # 4800, 6800
        pvnet = PolicyValueNetwork(config)
        checkpoint = torch.load(f'pvnet_v2_{i}.tar')
        pvnet.load_state_dict(checkpoint['model_state_dict'])
        agents.append(AZAgent2(pvnet, 100))

    res = [[None] * len(agents) for _ in agents]
    scores = copy.deepcopy(res)

    for i in range(len(res)):
        for j in range(len(res)):
            # print(i, j)
            if i != j:
                w, l, t = simulate_games(agents[i], agents[j], games)
                res[i][j] = (w, l, t)
                scores[i][j] = (w - l) / games
            else:
                scores[i][j] = 0

    print([sum(i) for i in scores])
    print(scores)
    print(res)