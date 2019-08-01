import numpy as np
import copy

class QLearningAgent():

    def __init__(self, alpha=.2, policy=None, gamma=.99, actions=None, observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action_id = None
        self.q_values = self._init_q_values()
        self.traning = True

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        """
            状態の初期化 
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self, q_values=None):
        if self.traning:
            action_id = self.policy.select_action(self.q_values[self.state])
        else:
            action_id = self.policy.select_greedy_action(self.q_values[self.state])
        self.previous_action_id = action_id
        action = self.actions[action_id]
        return action

    def observe(self, next_state, reward=None):
        """
            次の状態と報酬の観測 
        """
        next_state = str(next_state)
        if next_state not in self.q_values: # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if self.traning and reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)

    def learn(self, reward):
        """
            Q値の更新 
        """
        q = self.q_values[self.previous_state][self.previous_action_id] # Q(s, a)
        max_q = max(self.q_values[self.state]) # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.q_values[self.previous_state][self.previous_action_id] = q + (self.alpha * (reward + (self.gamma*max_q) - q))

