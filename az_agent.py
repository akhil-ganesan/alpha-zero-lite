import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from agent import Agent, simulate_games, RandomAgent, MiniMaxAgent, MCTSAgent


@dataclass
class Config(object):
    rn_filters: int = 16 # 256
    pol_filters: int = 4 # 32
    val_filters: int = 4 # 32
    val_hidden: int = 16 # 256
    layers: int = 2
    input_shape: tuple[int] = (2, 3, 3)
    actions: int = 9


class ResNetBlock(nn.Module):
    def __init__(self, filters, shortcut=False):
        super().__init__()
        self.resnet_body = nn.Sequential(
            nn.LazyConv2d(filters, kernel_size=3, stride=1, padding=1),
            # nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(filters, kernel_size=3, stride=1, padding=1),
            # nn.LazyBatchNorm2d()
        )

        if shortcut:
            self.shortcut = nn.LazyConv2d(filters, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = None

    def forward(self, x):
        y = self.resnet_body(x)
        if self.shortcut:
            x = self.shortcut(x)
        y += x
        return F.relu(y)


class PolicyValueNetwork(nn.Module):
    def __init__(self, config):
        # Input Shape is (C, H, W)
        super().__init__()
        self.resnet_body = nn.ModuleList([ResNetBlock(config.rn_filters,
                              config.rn_filters != config.input_shape[0]) for _ in range(config.layers)])
        self.policy_head = nn.Sequential(
            nn.LazyConv2d(config.pol_filters, kernel_size=3, stride=1, padding=1),
            # nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(config.actions),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.LazyConv2d(config.val_filters, kernel_size=3, stride=1, padding=1),
            # nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(config.val_hidden),
            nn.ReLU(),
            nn.LazyLinear(1),
            nn.Tanh()
        )
        # Note: Can add initial pooling to downsample board size for large games

    def forward(self, x):
        for b in self.resnet_body:
            x = b(x)
        return self.policy_head(x), self.value_head(x)


class AZAgent(Agent):
    def __init__(self, pvnet, simulations=100):
        self.simulations = simulations
        self.pvnet = pvnet


    class Node:
        def __init__(self, pvnet, game, value=None, p=1, action=None, noise=False, eps=0.25, alpha=(10/4.5)):
            self.pvnet, self.game, self.q, self.n, self.p = pvnet, game, 0, 0, p

            self.end = value is not None
            if not self.end:
                self.p_n, self.v = pvnet(game.get_perspective_state())
                self.p_n, self.v = self.p_n.reshape(-1), self.v.reshape(-1)
                # Add Dirichlet Noise
                if noise:
                    n = torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha] * len(self.p_n)))
                    self.p_n = self.p_n * (1 - eps) + eps * n.sample()

                # Mask illegal actions and rebalance
                valid_actions = torch.tensor(list(self.game.get_valid_moves()), dtype=torch.long)
                mask = torch.zeros(self.p_n.shape)
                mask[valid_actions] = 1
                self.p_n *= mask
                self.p_n /= self.p_n.sum()
                self.a_s = torch.argsort(self.p_n, descending=True)
            else:
                # From the perspective of the player playing on ended board, they have lost (-1)
                self.p_n, self.v = torch.empty(0), torch.tensor(value if not value else -1)
                self.a_s = torch.empty(0)

            self.adj = []
            self.action = action

            self.update(self.v)

        def expand(self):
            if self.is_leaf():
                a = self.a_s[len(self.adj)].item()
                gc = copy.deepcopy(self.game)
                _, val = gc.step(a)
                node = AZAgent.Node(self.pvnet, gc, value=val, p=self.p_n[a].item(), action=a)
                self.adj.append(node)
                return node
            return None

        def is_leaf(self):
            return not self.end and len(self.adj) < len(self.game.get_valid_moves())

        def update(self, g):
            self.q = ((self.n * self.q) + g)/(self.n + 1)
            self.n += 1

        def pUCT(self, c_puct=1.5):
            nt = sum(c.n for c in self.adj)
            return max(self.adj, key=lambda c: -c.q + c_puct * c.p * math.sqrt(nt) / (1 + c.n))

        def UCB(self, Qmax=1, Qmin=-1, c1=1.25, c2=19652):
            nt = sum(c.n for c in self.adj)
            return max(self.adj, key=lambda c: ((Qmax-c.q)/(Qmax-Qmin)) +
                                               c.p * math.sqrt(nt) / (1 + c.n) *
                                               (c1 + math.log((nt + c2 + 1) / c2)))


    def get_action_enhanced(self, game, temp=0):
        root = AZAgent.Node(self.pvnet, game, noise=(temp != 0))

        def dfs(node: AZAgent.Node):
            if not node.is_leaf() and not node.end:
                # Heuristic Minimax Selection
                g = dfs(node.pUCT())
            elif not node.end:
                leaf = node.expand()
                g = leaf.v
            else:
                g = -node.v
            g = -g
            node.update(g)
            return g

        for _ in range(self.simulations - 1):
            dfs(root)

        # Visit-Based Sampling
        pi = torch.zeros(root.p_n.shape, dtype=torch.float)
        valid_pi = []
        valid_actions = []
        for c in root.adj:
            pi[c.action] = c.n
            valid_actions.append(c.action)
            valid_pi.append(c.n)
        valid_pi = torch.tensor(valid_pi, dtype=torch.float)
        pi /= torch.sum(pi)
        valid_pi /= torch.sum(valid_pi)
        if not temp:
            return valid_actions[torch.argmax(valid_pi).item()], pi
        else:
            p = torch.pow(valid_pi, 1 / temp)
            p /= sum(p)
            return valid_actions[torch.multinomial(p, num_samples=1).item()], pi

    def get_action(self, game, player, temp=0, *args, **kwargs):
        return self.get_action_enhanced(game, temp)[0]


# Debugging a specific game
# if __name__ == "__main__":
#     game = TicTacToe()
#     moves = [4, 0, 2, 1, 8, 6] # 6, 7]
#     game.simulate(moves)
#     game.print_board()
#
#     config = Config()
#     pvnet = PolicyValueNetwork(config)
#     agent = AZAgent2(pvnet, 100)
#     print(agent.get_action(game, len(moves) % 2, temp=1))

# Running an array of games
if __name__ == "__main__":
    config = Config()
    pvnet = PolicyValueNetwork(config)
    simulate_games(AZAgent(pvnet), RandomAgent(), 20)
    simulate_games(AZAgent(pvnet), MCTSAgent(), 20)
    simulate_games(AZAgent(pvnet), MiniMaxAgent(), 3)

