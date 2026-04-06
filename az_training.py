import torch
import torch.nn.functional as F

from RLA.agent import simulate_games, RandomAgent, MiniMaxAgent, MCTSAgent
from RLA.env import RLEnv
from RLA.az_agent import AZAgent, PolicyValueNetwork, Config

EPOCHS = 800 # Each epoch is a game; this can be batched (with randomness in MCTS)

# Can ideally parallelize playing games (populating the buffer)
# & model training (learning from the replay data)

def play_game(network, temp=1, rounds=20):
    rl = RLEnv(AZAgent(network, rounds), AZAgent(network, rounds))
    states, w, actions = rl.az_play(temp)
    # print(actions)
    return states.float(), torch.tensor([w * (-1 if i % 2 else 1) for i in range(len(actions))]).float(), actions
    # return states.float(), torch.full(size=(len(actions),), fill_value=w).float(), actions



if __name__ == "__main__":
    config = Config()
    pvnet = PolicyValueNetwork(config)
    checkpoint = torch.load(f'pvnet_200.tar')
    pvnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(pvnet.parameters(), lr=1e-3) # 3e-4)

    device = "cpu"
    losses = []
    pvnet.train()

    for i in range(1, EPOCHS + 1):
        # pvnet.eval()
        with torch.no_grad():
            states, w, actions = play_game(pvnet, rounds=100)

        optimizer.zero_grad()
        states = states.to(device)
        w = w.to(device)
        actions = actions.to(device)

        # print(states[0])

        policy, values = pvnet(states) # pvnet(states[0].reshape(1, 2, 3, 3))
        # print(values)
        # print(actions)
        # print(policy)
        # print(values)
        # print(w)
        loss = F.cross_entropy(policy, actions) + F.mse_loss(values.reshape(-1), w)
        # F.mse_loss(values.reshape(-1), w[0].reshape(-1))
        # print(w[0])
        # print(loss)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(i)

        if i % 200 == 0:
            mean_loss = sum(losses) / len(losses)
            print(f"Loss: {mean_loss}")
            with torch.no_grad():
                simulate_games(AZAgent(pvnet, rounds=100), RandomAgent(), 10)
                simulate_games(AZAgent(pvnet, rounds=100), MiniMaxAgent(), 3)
                simulate_games(AZAgent(pvnet, rounds=100), MCTSAgent(rounds=100, rollouts=1), 10)

            torch.save({
                'epoch': i,
                'model_state_dict': pvnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
            }, f'pvnet_{i + 200}.tar')
            losses = []

