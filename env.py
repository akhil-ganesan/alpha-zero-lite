import torch

class TicTacToe:

    def __init__(self):
        # Position Arrangement:
        # 0, 1, 2
        # 3, 4, 5
        # 6, 7, 8

        self.open_pos = set(range(9))
        self.p1_pos = [0 for _ in range(9)]
        self.p2_pos = [0 for _ in range(9)]
        self.p1_m = 0
        self.p2_m = 0

        # Win Cache: 0, 1, 2 for rows; 3, 4, 5 for cols; 6 & 7 for diagonals
        self.p1 = [0 for _ in range(8)]
        self.p2 = [0 for _ in range(8)]

    def get_state(self):
        return torch.tensor([self.p1_pos, self.p2_pos], dtype=torch.float).reshape(1, 2, 3, 3)

    def get_perspective_state(self):
        return torch.tensor([self.p1_pos, self.p2_pos] if self.get_turn() else
                            [self.p2_pos, self.p1_pos], dtype=torch.float).reshape(1, 2, 3, 3)

    def get_valid_moves(self):
        return self.open_pos

    def get_turn(self):
        # True for P1, False for P2
        return self.p1_m == self.p2_m

    def step(self, action):
        if action not in self.open_pos:
            print("Invalid action")
            return False, None
        if self.get_turn():
            pos, p, player = self.p1_pos, self.p1, 1
            self.p1_m += 1
        else:
            pos, p, player = self.p2_pos, self.p2, -1
            self.p2_m += 1
        self.open_pos.remove(action)
        pos[action] = 1
        checks = [(True, action // 3), (True, (action % 3) + 3),
                (action // 3 == action % 3, -2),
                (action // 3 + action % 3 == 2, -1)]

        for cond, idx in checks:
            if cond:
                p[idx] += 1
                if p[idx] == 3:
                    return True, player

        end = self.p1_m + self.p2_m == 9
        return end, None if not end else 0

    def print_board(self):
        for i in range(3):
            for j in range(3):
                n = 3*i + j
                print("X" if self.p1_pos[n] else ("O" if self.p2_pos[n] else "?"), end=" ")
            print()

    def simulate(self, actions):
        for a in actions:
            self.step(a)

class RLEnv:
    def __init__(self, agent1, agent2):
        self.game = TicTacToe()
        self.agents = (agent1, agent2)

    def play(self):
        end, t, w = False, -1, None
        while not end:
            t += 1
            player = self.agents[t % 2]
            if player:
                end, w = self.game.step(player.get_action(self.game, t % 2))

        # print("X Win" if w == 1 else ("O Win" if w == -1 else "Draw"))
        # self.game.print_board()
        return w


    def supervised_az_play(self, temp=1, az=(True, True)):
        end, t, w = False, -1, None
        states = []
        actions = []
        while not end:
            t += 1
            states.append(self.game.get_perspective_state())
            player = self.agents[t % 2]
            if az[t % 2]:
                action, pi = player.get_action_enhanced(self.game, temp=temp)
            else:
                action, pi = player.get_action(self.game, t % 2), torch.zeros(9, dtype=torch.float)
            actions.append(pi)
            end, w = self.game.step(action)

        return torch.cat(states, dim=0), w, torch.stack(actions)