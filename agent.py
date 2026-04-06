import copy
import math
import random
from abc import ABC, abstractmethod

from RLA.env import TicTacToe, RLEnv


class Agent(ABC):

    @abstractmethod
    def get_action(self, game, player, *args, **kwargs):
        pass


class ManualAgent(Agent):
    def get_action(self, game, player, *args, **kwargs):
        game.print_board()
        while True:
            action = input("Enter action (position to place): ")
            try:
                action = int(action)
                if action in game.get_valid_moves():
                    return action
                else:
                    print("Invalid action")
            except ValueError:
                print("Invalid action")


class RandomAgent(Agent):
    def get_action(self, game, player, *args, **kwargs):
        return random.choice(list(game.get_valid_moves()))


class MiniMaxAgent(Agent):
    def __init__(self, rmt=9):
        self.rmt = rmt

    def get_action(self, game, player, *args, **kwargs):

        # Random starting move to test
        if len(game.get_valid_moves()) > self.rmt:
            return random.choice(list(game.get_valid_moves()))

        def dfs(p1_pos=game.p1_pos.copy(),
                p2_pos=game.p2_pos.copy(),
                p1_m=game.p1_m,
                p2_m=game.p2_m,
                p1=game.p1.copy(),
                p2=game.p2.copy(),
                actions=game.get_valid_moves().copy(),
                maximize=(player == 0),
                alpha=-1,
                beta=1):

            value = 1 if (3 in p1) else (-1 if 3 in p2 else (0 if p1_m + p2_m == 9 else None))

            if value is not None:
                return None, value

            a_optim, v_optim = None, None

            if maximize:
                pos, p = p1_pos, p1
            else:
                pos, p = p2_pos, p2

            for action in actions.copy():
                # Apply the action
                actions.remove(action)
                pos[action] = 1
                for cond, idx in [(True, action // 3), (True, (action % 3) + 3),
                          (action // 3 == action % 3, -2),
                          (action // 3 + action % 3 == 2, -1)]:
                    if cond:
                        p[idx] += 1

                _, v = dfs(p1_pos, p2_pos, p1_m + (1 if max else 0),
                           p2_m + (0 if max else 1), p1, p2, actions, not maximize, alpha, beta)

                if v_optim is None or (maximize and v > v_optim) or (not maximize and v < v_optim):
                    a_optim = action
                    v_optim = v
                    if maximize:
                        alpha = max(alpha, v_optim)
                    else:
                        beta = min(beta, v_optim)

                # Undo the action
                actions.add(action)
                pos[action] = 0
                for cond, idx in [(True, action // 3), (True, (action % 3) + 3),
                                  (action // 3 == action % 3, -2),
                                  (action // 3 + action % 3 == 2, -1)]:
                    if cond:
                        p[idx] -= 1

                # AB Pruning
                if (not maximize and v_optim <= alpha) or (maximize and v_optim >= beta):
                   break

            return a_optim, v_optim

        return dfs()[0]


class MCTSAgent(Agent):


    class Node:
        def __init__(self, game, value=None, rollouts=1, action=None):
            self.game = game
            self.end = value is not None
            if self.end:
                self.u = value*(rollouts + 1) # +1 tiebreak for terminal states
            else:
                self.u = self.simulate_random(rollouts)
            self.rollouts = rollouts
            self.n = rollouts
            self.q = game.get_valid_moves().copy() if not self.end else []
            self.adj = []
            self.action = action

        def expand(self):
            u = n = 0
            if not self.end and self.q:
                action = self.q.pop()
                gc = copy.deepcopy(self.game)
                _, val = gc.step(action)
                self.adj.append(MCTSAgent.Node(gc, value=val, rollouts=self.rollouts, action=action))
                u = self.adj[-1].u
                n = self.adj[-1].n
            return u, n

        def simulate_random(self, rollouts):
            u = 0
            for _ in range(rollouts):
                game = copy.deepcopy(self.game)
                end, val = False, None
                while not end:
                    end, val = game.step(random.choice(list(game.get_valid_moves())))
                u += val
            return u

        def update(self, u, n):
            self.u += u
            self.n += n

        def UCB_h(self, p_n, c=math.sqrt(2)):
            return self.selection_h() + c*math.sqrt(math.log(p_n) / self.n)

        def selection_h(self):
            multiplier = (1, -1)[self.game.get_turn()]
            return self.u*multiplier/self.n


    def __init__(self, rounds=100, rollouts=1):
        self.rounds = rounds
        self.rollouts = rollouts


    def get_action(self, game, player, *args, **kwargs):
        root = MCTSAgent.Node(copy.deepcopy(game), rollouts=self.rollouts)

        def dfs(node: MCTSAgent.Node):
            u = n = 0
            if not node.q and not node.end:
                # UCB Minimax Selection
                if not node.adj:
                    node.game.print_board()
                    print(node.n, node.u, node.end, node.adj)
                u, n = dfs(max(node.adj, key=lambda c: c.UCB_h(node.n)))
            elif node.q:
                u, n = node.expand()
            else:
                u, n = node.u, self.rollouts
            node.update(u, n)
            return u, n

        for _ in range(self.rounds-1):
            dfs(root)

        # for c in root.adj:
        #     print(c.action)
        #     c.game.print_board()
        #     print(c.u, c.n)
        #     print(c.selection_h())
        #     print(c.UCB_h(root.n))
        #     print()

        return max(root.adj, key=lambda c: c.selection_h()).action

        # print(root.u)
        # print(root.n)
        # print("Expanding")
        # u, n = root.expand()
        # print(u)
        # print(n)
        # root.adj[0].game.print_board()


def simulate_games(agent1, agent2, games):
    wins, losses, ties = 0, 0, 0
    for _ in range(games):
        rl = RLEnv(agent1, agent2)
        w = rl.play()
        if w == 1:
            wins += 1
        elif w == -1:
            losses += 1
        else:
            ties += 1

    print(f"Agent 1 Wins/Losses/Ties: {wins}/{losses}/{ties} ({games} total games)")
    return wins, losses, ties


if __name__ == "__main__":
    simulate_games(MCTSAgent(rounds=300, rollouts=1), MCTSAgent(rounds=300, rollouts=1), 10)

    # print("Random")
    # simulate_games(MCTSAgent(), RandomAgent(), 20)
    #
    # print("20")
    # simulate_games(MCTSAgent(rounds=20, rollouts=1), MiniMaxAgent(), 20)
    # simulate_games(MCTSAgent(rounds=20, rollouts=1), MiniMaxAgent(7), 20)
    # print("150")
    # simulate_games(MCTSAgent(rounds=150, rollouts=1), MiniMaxAgent(), 20)
    # simulate_games(MCTSAgent(rounds=150, rollouts=1), MiniMaxAgent(7), 20)
    # print("300")
    # simulate_games(MCTSAgent(rounds=150, rollouts=1), MiniMaxAgent(), 20)
    # simulate_games(MCTSAgent(rounds=150, rollouts=1), MiniMaxAgent(7), 20)

    # simulate_games(MCTSAgent(rounds=3, rollouts=3), MCTSAgent(rounds=100, rollouts=10), 20)
    # simulate_games(RandomAgent(), RandomAgent(), 10)
    # simulate_games(RandomAgent(), MiniMaxAgent(), 10)
    # simulate_games(MiniMaxAgent(), RandomAgent(), 10)
    # simulate_games(MiniMaxAgent(), MiniMaxAgent(), 10)


# Debugging specific games

# if __name__ == "__main__":
#     game = TicTacToe()
#     moves = [4, 0, 2, 1, 8, 6]
#     game.simulate(moves)
#     game.print_board()
#     agent = MCTSAgent()
#     print(agent.get_action(game, len(moves) % 2))

# if __name__ == "__main__":
#     game = TicTacToe()
#     # moves = [8, 3, 7] # 1, 6, 7, 8
#     moves = [4, 0, 2, 1, 8]
#     game.simulate(moves)
#     game.print_board()
#     agent = MiniMaxAgent()
#     print(agent.get_action(game, len(moves) % 2))