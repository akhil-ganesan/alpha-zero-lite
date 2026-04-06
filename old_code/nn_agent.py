import copy
import math
import random

import torch
from torch import nn
import torch.nn.functional as F

from RLA.agent import RandomAgent
from RLA.env import TicTacToe
from agent import Agent, simulate_games, MCTSAgent


# class H():
#     def __call__(self, *args, **kwargs):
#         return random.randint(1, 100)
#
# class F:
#     def __call__(self, *args, **kwargs):
#         return torch.softmax(torch.rand(9), dim=-1), random.random()
#
# class G:
#     def __call__(self, *args, **kwargs):
#         return random.randint(1, 100), random.random()

class RepresentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9, 9)
        )

    def forward(self, x):
        return self.model(x)


class PredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 9*4),
            nn.ReLU()
        )
        self.p_n = nn.Linear(9*4, 9)
        self.v = nn.Linear(9*4, 1)

    def forward(self, x):
        x = self.model(x)
        return self.p_n(x), self.v(x)


class DynamicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()


class MuZeroAgent(Agent):
    def __init__(self, h, f, g, rounds=20):
        self.rounds = rounds
        self.h = h
        self.f = f
        self.g = g

    class Node:
        def __init__(self, agent, s, r=0, p=1, action=0):
            self.agent = agent
            self.s = s
            self.q = 0
            self.n = 0
            self.r = r
            self.p = p
            self.p_n, self.v = agent.f(s)
            self.a_s = torch.argsort(self.p_n, descending=True)
            self.adj = []
            self.action = action

        def expand(self):
            if self.is_leaf():
               a = self.a_s[len(self.adj)]
               s1, r1 = self.agent.g(torch.stack((self.s, F.one_hot(a, num_classes=9)), dim=-1)) #
               node = MuZeroAgent.Node(self.agent, s1, r1, self.p_n[a].item(), a.item())
               self.adj.append(node)
               return node

            return None

        def is_leaf(self):
            return len(self.adj) < len(self.p_n)

        def update(self, g):
            self.q = ((self.n * self.q) + g)/(self.n + 1)
            self.n += 1

        def UBC_h(self, p1, Qmax=1, Qmin=0, c1=1.25, c2=19652):
            nb = sum(node.n for node in self.adj)
            return max(self.adj, key=lambda node: (((node.q - Qmin) if p1 else (Qmax - node.q))/(Qmax - Qmin)) +
                                                  (node.p * math.sqrt(nb / (1 + node.n)) *
                                                   (c1 + math.log((nb + c2 + 1) / c2)))) if self.adj else self

    def get_action(self, game, player, gamma=1, *args, **kwargs):
        s0 = self.h(torch.tensor([game.p1_pos, game.p2_pos]).reshape(2, 3, 3))
        root = MuZeroAgent.Node(self, s0)
        root.update(root.v)
        q_bound = [root.v, root.v]

        def dfs(node: MuZeroAgent.Node, p1=game.get_turn(), q_bound=q_bound):
            if not node.is_leaf():
                # UCB Minimax Selection
                g = dfs(node.UBC_h(p1, q_bound[0], q_bound[1]), not p1, q_bound)
            else:
                leaf = node.expand()
                # Modified discounted rewards formula
                g = leaf.v * gamma + leaf.r
                leaf.update(g)
                q_bound[0] = max(q_bound[0], leaf.q)
                q_bound[1] = min(q_bound[1], leaf.q)

            g = g * gamma + node.r
            node.update(g)
            q_bound[0] = max(q_bound[0], node.q)
            q_bound[1] = min(q_bound[1], node.q)

            return g

        for _ in range(self.rounds-1):
            dfs(root)

        # Greedy Selection
        valid_actions = game.get_valid_moves()
        return max(root.adj, key=lambda c: c.q * (-1, 1)[game.get_turn()] if c.action in valid_actions else -math.inf).action


class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(4*4, 4*4*4),
            nn.GELU()
        )
        self.policy = nn.Linear(4*4*4, 9)
        self.value = nn.Sequential(nn.Linear(4*4*4, 1), nn.Tanh())

    def forward(self, x):
        x = self.model(x)
        return self.policy(x), self.value(x)


class AlphaZeroAgent(Agent):


    def __init__(self, pvnet, device='cpu', rounds=20):
        self.rounds = rounds
        self.pvnet = pvnet
        self.device = device


    class Node:
        def __init__(self, game, pvnet, device, value=None, action=None):
            self.game = game
            self.pvnet = pvnet
            self.end = value is not None
            if self.end:
                self.policy, self.u = None, value # +1 tiebreak for terminal states
            else:
                self.policy, self.u = pvnet(game.get_state().to(device))
                self.u = self.u.item()
                self.policy = self.policy.tolist()
            self.n = 1
            self.q = game.get_valid_moves().copy() if not self.end else []
            self.adj = []
            self.action = action
            self.device = device

        def expand(self):
            u = n = 0
            if not self.end and self.q:
                action = self.q.pop()
                gc = copy.deepcopy(self.game)
                _, val = gc.step(action)
                self.adj.append(AlphaZeroAgent.Node(gc, self.pvnet, self.device, value=val, action=action))
                u = self.adj[-1].u
                n = self.adj[-1].n
            return u, n

        def update(self, u, n):
            self.u += u
            self.n += n

        def UCB_h(self, p_n, c=math.sqrt(2)):
            return self.selection_h() + c*math.sqrt(math.log(p_n) / self.n)

        def selection_h(self):
            multiplier = (1, -1)[self.game.get_turn()]
            return self.u*multiplier/self.n


    def get_action(self, game, player, *args, **kwargs):
        root = AlphaZeroAgent.Node(copy.deepcopy(game), self.pvnet, self.device)

        def dfs(node: AlphaZeroAgent.Node):
            u = n = 0
            if not node.q and not node.end:
                # UCB Minimax Selection
                u, n = dfs(max(node.adj, key=lambda c: c.UCB_h(node.n)))
            elif node.q:
                u, n = node.expand()
            node.update(u, n)
            return u, n

        for _ in range(self.rounds-1):
            dfs(root)

        # def check(node=root):
        #     if node:
        #         print()
        #         node.game.print_board()
        #         print(node.u, node.n)
        #         for n in node.adj:
        #             check(n)

        # check()

        return root.adj[torch.multinomial(
            F.softmax(torch.tensor([c.selection_h() for c in root.adj]), dim=-1), 1).item()].action
        # return max(root.adj, key=lambda c: c.selection_h()).action


if __name__ == "__main__":
    game = TicTacToe()
    moves = [4, 0, 2] # , 2, 1] #, 8, 6]
    game.simulate(moves)
    game.print_board()
    f = PolicyValueNetwork()
    agent = AlphaZeroAgent(f, rounds=9)
    print(agent.get_action(game, len(moves) % 2))


# if __name__ == "__main__":
#     game = TicTacToe()
#     moves = [4, 0, 2, 1, 8, 6]
#     game.simulate(moves)
#     game.print_board()
#     h, f, g = H(), F(), G()
#     agent = MuZeroAgent(h, f, g, 9)
#     print(agent.get_action(game, len(moves) % 2))

# if __name__ == "__main__":
    # h, f, g = H(), F(), G()
    # simulate_games(MCTSAgent(), MuZeroAgent(h, f, g), 20)
    # simulate_games(RandomAgent(), MuZeroAgent(h, f, g), 5)
    # simulate_games(MuZeroAgent(h, f, g), MuZeroAgent(h, f, g), 1)
    # pvnet = PolicyValueNetwork()
    # simulate_games(RandomAgent(), AlphaZeroAgent(pvnet), 20)
    # simulate_games(AlphaZeroAgent(pvnet), AlphaZeroAgent(pvnet), 20)