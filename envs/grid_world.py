import numpy as np
import copy

FILED_TIPE = {
    "N":0, 
    "G":2, 
    }

ACTIONS = {
    "UP": 0, 
    "DOWN": 1, 
    "LEFT": 2, 
    "RIGHT":3
    }

class GridWorld():
    """
        グリッドワールド
    """
    def __init__(self):


        self.map = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                ]

        self.start_pos = 3, 0 # エージェントのスタート地点(y, x)
        self.agent_pos = copy.deepcopy(self.start_pos)  # エージェントがいる地点

    def step(self, action):
        """
            return pos, reward
        """
        to_y, to_x = self.agent_pos
        # 移動可能かどうかの確認。移動不可能であれば、ポジションはそのままにマイナス報酬@todo
        if self._is_possible_action(to_x, to_y, action) == False:
            return self.agent_pos, -1, False

        if action == ACTIONS["UP"]:
            to_y += -1
        elif action == ACTIONS["DOWN"]:
            to_y += 1
        elif action == ACTIONS["LEFT"]:
            to_x += -1
        elif action == ACTIONS["RIGHT"]:
            to_x += 1

        is_goal = self._is_goal(to_x, to_y) # ゴールしているかの確認
        reward = self._compute_reward(to_x, to_y)
        self.agent_pos = to_y, to_x
        return self.agent_pos, reward, is_goal

    def _is_goal(self, x, y):
        """
            x, yがゴール地点かの判定 
        """
        if self.map[y][x] == FILED_TIPE["G"]:
            return True
        else:
            return False

    def _is_possible_action(self, x, y, action):
        """ 
            実行可能な行動かどうかの判定
        """
        to_x = x
        to_y = y

        if action == ACTIONS["UP"]:
            to_y += -1
        elif action == ACTIONS["DOWN"]:
            to_y += 1
        elif action == ACTIONS["LEFT"]:
            to_x += -1
        elif action == ACTIONS["RIGHT"]:
            to_x += 1
        else:
            raize("Action Eroor")

        if len(self.map) <= to_y or 0 > to_y:
            return False
        elif len(self.map[0]) <= to_x or 0 > to_x:
            return False

        return True

    def _compute_reward(self, x, y):
        if self.map[y][x] == FILED_TIPE["N"]:
            return -1
        elif self.map[y][x] == FILED_TIPE["G"]:
            return 100

    def reset(self):
        self.agent_pos = self.start_pos
        return self.start_pos
