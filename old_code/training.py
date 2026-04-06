import torch
import torch.nn.functional as F

from RLA.agent import simulate_games, MCTSAgent
from RLA.env import RLEnv
from RLA.old_code.nn_agent import AlphaZeroAgent, PolicyValueNetwork

EPOCHS = 1000 # Each epoch is a game; this can be batched (with randomness in MCTS)
# states_replay_buffer = []
# action_replay_buffer = []
# value_replay_buffer = []

# Can ideally parallelize playing games (populating the buffer)
# & model training (learning from the replay data)

def play_game(network):
    rl = RLEnv(AlphaZeroAgent(network), AlphaZeroAgent(network)) # AlphaZeroAgent(network))
    states, w, actions = rl.alpha_zero_play()
    # print(f"W: {w}")
    return states.float(), torch.full(size=(len(actions),), fill_value=w).float(), actions
    # action_replay_buffer.append(actions)
    # for i in range(actions.size(0)):
    #     value_replay_buffer.append(w)

    # print(states, w, actions)
    # print(states.shape, actions.shape)



if __name__ == "__main__":

    pvnet = PolicyValueNetwork()
    optimizer = torch.optim.AdamW(pvnet.parameters(), lr=1e-5)

    device = "cpu" # torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # if device == "xpu":
    #     import intel_extension_for_pytorch as ipex
    #
    #     # pvnet, optimizer = ipex.optimize(pvnet, dtype=torch.bfloat16, optimizer=optimizer)
    # elif device == "cuda":
    #     pvnet = torch.compile(pvnet)
    #     torch.set_float32_matmul_precision('high')

    for i in range(EPOCHS):
        pvnet.eval()
        states, w, actions = play_game(pvnet)

        pvnet.train()
        optimizer.zero_grad()
        # with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        states = states.to(device)
        w = w.to(device)
        actions = actions.to(device)

        policy, values = pvnet(states)
        # print(actions)
        # print(policy.shape, values.shape, actions.shape, w.shape)
        loss = F.cross_entropy(policy, actions) + F.mse_loss(values.reshape(-1), w)
        # print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            pvnet.eval()
            # simulate_games(AlphaZeroAgent(pvnet), RandomAgent(), 10)
            # simulate_games(AlphaZeroAgent(pvnet), MiniMaxAgent(), 10)
            simulate_games(AlphaZeroAgent(pvnet, rounds=4), MCTSAgent(rollouts=4, rounds=4), 10)


        # s = torch.cat(states_replay_buffer, dim=0)
        # a = torch.cat(action_replay_buffer, dim=0)
        # v = torch.tensor(value_replay_buffer).float()

        # # Reset Buffers
        # states_replay_buffer = []
        # action_replay_buffer = []
        # value_replay_buffer = []








