import time

import torch
import torch.nn.functional as F

from agent import simulate_games, RandomAgent, MiniMaxAgent, MCTSAgent
from az_agent import AZAgent
from env import RLEnv
from az_agent import PolicyValueNetwork, Config

EPOCHS = 4000

def play_game(network, temp=1.0, simulations=100):
    rl = RLEnv(AZAgent(network, simulations), AZAgent(network, simulations))
    states, w, actions = rl.supervised_az_play(temp)
    return states.float(), torch.tensor([w * (-1 if i % 2 else 1) for i in range(len(actions))]).float(), actions

if __name__ == "__main__":
    t1 = time.time()
    config = Config()
    pvnet = PolicyValueNetwork(config)
    # checkpoint = torch.load(f'pvnet_v2_1000.tar')
    # pvnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(pvnet.parameters(), lr=1e-3) # 3e-4

    losses = []
    pvnet.train()

    for i in range(EPOCHS + 1):
        with torch.no_grad():
            # Linearly-decaying temperature from 1 to 0.25
            temp = ((EPOCHS - i*0.75)/EPOCHS)
            states, w, actions = play_game(pvnet, temp=temp)

        optimizer.zero_grad()
        policy, values = pvnet(states)
        loss = F.cross_entropy(policy, actions) + F.mse_loss(values.reshape(-1), w)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(i)

        if i % 1000 == 0:
            mean_loss = sum(losses) / len(losses)
            print(f"Loss: {mean_loss}")
            with torch.no_grad():
                simulate_games(AZAgent(pvnet), RandomAgent(), 10)
                simulate_games(AZAgent(pvnet), MiniMaxAgent(), 3)
                simulate_games(AZAgent(pvnet), MCTSAgent(), 10)

            torch.save({
                'epoch': i,
                'model_state_dict': pvnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
            }, f'pvnet_{i}.tar')
            losses = []

    print(f"Runtime: {time.time() - t1}")

