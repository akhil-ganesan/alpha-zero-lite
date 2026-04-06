import copy
import math

import torch

from RLA.agent import Agent, simulate_games, RandomAgent, MiniMaxAgent, MCTSAgent
from RLA.az_agent import Config, PolicyValueNetwork
from RLA.env import TicTacToe

class AZAgent2(Agent):
    def __init__(self, pvnet, simulations=100):
        self.simulations = simulations
        self.pvnet = pvnet


    class Node:
        def __init__(self, pvnet, game, value=None, p=1, action=None, noise=False, eps=0.25, alpha=0.3):
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
                node = AZAgent2.Node(self.pvnet, gc, value=val, p=self.p_n[a].item(), action=a)
                self.adj.append(node)
                return node
            return None

        def is_leaf(self):
            return not self.end and len(self.adj) < len(self.game.get_valid_moves())

        def update(self, g):
            self.q = ((self.n * self.q) + g)/(self.n + 1)
            self.n += 1

        def UCB(self, c_puct=1.5):
            nt = sum(c.n for c in self.adj)
            return max(self.adj, key=lambda c: -c.q + c_puct * c.p * math.sqrt(nt) / (1 + c.n))


    def get_action_enhanced(self, game, temp=0):
        root = AZAgent2.Node(self.pvnet, game, noise=(temp != 0))

        def dfs(node: AZAgent2.Node):
            if not node.is_leaf() and not node.end:
                # UCB Minimax Selection
                g = dfs(node.UCB())
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

        def check(node=root):
            if node:
                print()
                node.game.print_board()
                print(node.v, node.p_n, node.n, node.q)
                for n in node.adj:
                    check(n)

        # check()

        # Visit-Based Sampling
        pi = torch.zeros(root.p_n.shape, dtype=torch.float)
        for c in root.adj:
            pi[c.action] = c.n
        pi /= torch.sum(pi)
        if not temp:
            return torch.argmax(pi).item(), pi
        else:
            p = torch.pow(pi, 1 / temp)
            p /= sum(p)
            return torch.multinomial(p, num_samples=1).item(), pi

    def get_action(self, game, player, temp=0, *args, **kwargs):
        return self.get_action_enhanced(game, temp)[0]



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

# if __name__ == "__main__":
#     config = Config()
#     pvnet = PolicyValueNetwork(config)
#     simulate_games(AZAgent2(pvnet), RandomAgent(), 20)
#     simulate_games(AZAgent2(pvnet), MCTSAgent(rounds=100, rollouts=1), 20)
#     simulate_games(AZAgent2(pvnet), MiniMaxAgent(), 3)

