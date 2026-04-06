import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from RLA.agent import Agent, simulate_games, RandomAgent, MiniMaxAgent, MCTSAgent
from RLA.env import TicTacToe


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
    def __init__(self, pvnet, rounds=20):
        self.rounds = rounds
        self.pvnet = pvnet


    class Node:
        def __init__(self, pvnet, game, value=None, p=1, action=0, p1=True):
            self.pvnet = pvnet
            self.game = game
            self.q = 0
            self.n = 0
            self.p = p

            self.end = value is not None
            if not self.end:
                self.p_n, self.v = pvnet(game.get_perspective_state()) # game.get_state() # F.softmax(torch.rand((9,)), dim=-1), 0 #
                self.p_n = self.p_n.reshape(-1)
                self.v = self.v.reshape(-1)
                # print(self.p_n, self.v)
                # Mask illegal actions and rebalance
                valid_indices = torch.tensor(list(self.game.get_valid_moves()), dtype=torch.long)
                mask = torch.zeros(self.p_n.shape)
                mask[valid_indices] = 1
                self.p_n *= mask
                self.p_n /= self.p_n.sum()
                self.a_s = torch.argsort(self.p_n, descending=True)
            else:
                self.p_n, self.v = [], torch.tensor(value if not value else -1)
                # From the perspective of the player playing on end board, they have lost (-1)
                # torch.tensor(value * (1 if p1 else -1))
                self.a_s = []

            self.adj = []
            self.action = action

            self.update(self.v)

        def expand(self, p1=True):
            if self.is_leaf():
                a = self.a_s[len(self.adj)].item()
                gc = copy.deepcopy(self.game)
                _, val = gc.step(a)
                node = AZAgent.Node(self.pvnet, gc, value=val, p=self.p_n[a].item(), action=a, p1=p1)
                self.adj.append(node)
                return node
            return None

        def is_leaf(self):
            return not self.end and len(self.adj) < len(self.game.get_valid_moves())

        def update(self, g):
            self.q = ((self.n * self.q) + g)/(self.n + 1)
            self.n += 1

        def nb(self, temp=1):
            return sum(math.pow(node.n, 1/temp) for node in self.adj)

        def UBC_h(self, p1, Qmax=1, Qmin=-1, c1=1.25, c2=19652):
            nb = self.nb()
            return max(self.adj, key=lambda node: ((Qmax - node.q) / (Qmax - Qmin)) +
                                                   (node.p * math.sqrt(nb / (1 + node.n)) *
                                                    (c1 + math.log((nb + c2 + 1) / c2))))

            # print([(node.game.open_pos, ((((node.q - Qmin) if p1 else (Qmax - node.q))/(Qmax - Qmin)) +
            #                                       (node.p * math.sqrt(nb / (1 + node.n)) *
            #                                        (c1 + math.log((nb + c2 + 1) / c2))))) for node in self.adj])
            # return max(self.adj, key=lambda node: ((((node.q - Qmin) if p1 else (Qmax - node.q))/(Qmax - Qmin)) +
            #                                       (node.p * math.sqrt(nb / (1 + node.n)) *
            #                                        (c1 + math.log((nb + c2 + 1) / c2)))))


    def get_action_enhanced(self, game, player=True, temp=0):
        player = game.get_turn()
        root = AZAgent.Node(self.pvnet, game, p1=player)
        # root.update(root.v)
        # print(root.p_n)
        # print(game.open_pos, root.v)

        def dfs(node: AZAgent.Node, even=True): # game.get_turn()
            if not node.is_leaf() and not node.end:
                # UCB Minimax Selection
                g = dfs(node.UBC_h(even), not even)
            elif not node.end:
                leaf = node.expand(p1=player)
                g = leaf.v
            else:
                g = -node.v

            node.update(-g)
            return -g

        for _ in range(self.rounds - 1):
            dfs(root)

        def check(node=root):
            if node:
                print()
                node.game.print_board()
                print(node.v, node.p_n, node.n, node.q)
                for n in node.adj:
                    check(n)

        # check()

        # Visit-Based Sampling (can add Dirichlet Noise)
        t = temp if temp else 1
        nb = root.nb(t)
        pi = torch.zeros(root.p_n.shape)
        for c in root.adj:
            pi[c.action] = math.pow(c.n, 1 / t) / nb

        return torch.multinomial(pi, num_samples=1).item() if temp else torch.argmax(pi).item(), pi

        # return root.adj[torch.multinomial(
        #     torch.tensor([(math.pow(c.n, 1 / temp) / nb) for c in root.adj]), num_samples=1).item()].action

        # Greedy Value Selection
        # valid_actions = game.get_valid_moves()
        # return max(root.adj, key=lambda c: c.q * (-1, 1)[game.get_turn()] if c.action in valid_actions else -math.inf).action


    def get_action(self, game, player, temp=0, *args, **kwargs):
        return self.get_action_enhanced(game, player, temp)[0]



# if __name__ == "__main__":
#     game = TicTacToe()
#     moves = [4, 0, 2, 1, 8] # 6, 7]
#     game.simulate(moves)
#     game.print_board()
#
#     config = Config()
#     pvnet = PolicyValueNetwork(config)
#     agent = AZAgent(pvnet, 10)
#     print(agent.get_action(game, len(moves) % 2))

if __name__ == "__main__":
    config = Config()
    pvnet = PolicyValueNetwork(config)
    # simulate_games(AZAgent(pvnet, rounds=300), AZAgent(pvnet, rounds=300), 10)
    simulate_games(AZAgent(pvnet, rounds=100), RandomAgent(), 20)
    simulate_games(AZAgent(pvnet, rounds=100), MCTSAgent(rounds=100, rollouts=1), 20)
    simulate_games(AZAgent(pvnet, rounds=100), MiniMaxAgent(), 3)
    # simulate_games(AZAgent(pvnet), AZAgent(pvnet), 20)

