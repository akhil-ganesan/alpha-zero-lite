import torch
import torch.nn.functional as F

from RLA.agent import simulate_games, RandomAgent, MiniMaxAgent, MCTSAgent
from RLA.cleaned_az_agent import AZAgent2
from RLA.env import RLEnv
from RLA.az_agent import PolicyValueNetwork, Config

EPOCHS = 4000

def play_game(network, temp=1, simulations=100):
    rl = RLEnv(AZAgent2(network, simulations), AZAgent2(network, simulations))
    states, w, actions = rl.cleaned_play(temp)
    return states.float(), torch.tensor([w * (-1 if i % 2 else 1) for i in range(len(actions))]).float(), actions

if __name__ == "__main__":
    config = Config()
    pvnet = PolicyValueNetwork(config)
    checkpoint = torch.load(f'pvnet_v2_200.tar')
    pvnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(pvnet.parameters(), lr=1e-3) # 3e-4)

    device = "cpu"
    losses = []
    pvnet.train()

    for i in range(1, EPOCHS + 1):
        with torch.no_grad():
            states, w, actions = play_game(pvnet)

        optimizer.zero_grad()
        policy, values = pvnet(states)
        loss = F.cross_entropy(policy, actions) + F.mse_loss(values.reshape(-1), w)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(i)

        if i % 200 == 0:
            mean_loss = sum(losses) / len(losses)
            print(f"Loss: {mean_loss}")
            with torch.no_grad():
                simulate_games(AZAgent2(pvnet), RandomAgent(), 10)
                simulate_games(AZAgent2(pvnet), MiniMaxAgent(), 3)
                simulate_games(AZAgent2(pvnet), MCTSAgent(rounds=100, rollouts=1), 10)

            torch.save({
                'epoch': i,
                'model_state_dict': pvnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
            }, f'pvnet_v2_{i + 4000}.tar')
            losses = []

