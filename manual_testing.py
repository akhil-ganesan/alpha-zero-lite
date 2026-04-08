import torch

from agent import ManualAgent, simulate_games, MiniMaxAgent, MCTSAgent
from az_agent import Config, PolicyValueNetwork
from az_agent import AZAgent

if __name__ == "__main__":
    config = Config()
    pvnet1 = PolicyValueNetwork(config)
    checkpoint = torch.load(f'pvnet_v4_4000.tar')
    pvnet1.load_state_dict(checkpoint['model_state_dict'])
    agent1 = AZAgent(pvnet1)
    pvnet2 = PolicyValueNetwork(config)
    checkpoint = torch.load(f'pvnet_v4_0.tar')
    pvnet2.load_state_dict(checkpoint['model_state_dict'])
    agent2 = AZAgent(pvnet2)

    # Testing against minimax
    rmt = 7
    w1, l1, _ = simulate_games(agent1, MiniMaxAgent(rmt=rmt), 20)
    w2, l2, _ = simulate_games(MiniMaxAgent(rmt=rmt), agent1, 20)
    print("Agent 1 Score ", {w1+l2-l1-w2})
    w1, l1, _ = simulate_games(agent2, MiniMaxAgent(rmt=rmt), 20)
    w2, l2, _ = simulate_games(MiniMaxAgent(rmt=rmt), agent2, 20)
    print("Agent 2 Score ", {w1+l2-l1-w2})

    # Testing against MCTS
    w1, l1, _ = simulate_games(agent1, MCTSAgent(), 20)
    w2, l2, _ = simulate_games(MCTSAgent(), agent1, 20)
    print("Agent 1 Score ", {w1+l2-l1-w2})
    w1, l1, _ = simulate_games(agent2, MCTSAgent(), 20)
    w2, l2, _ = simulate_games(MCTSAgent(), agent2, 20)
    print("Agent 2 Score ", {w1+l2-l1-w2})

    simulate_games(agent1, ManualAgent(), 1)