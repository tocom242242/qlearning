import numpy as np
import copy

class QLearningAgent():
    """
       qlearning エージェント
    """
    def __init__(self, alpha=0.2, policy=None, gamma=0.99, actions=None, observation=None, alpha_decay_rate=None, epsilon_decay_rate=None):
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.name = "qlearning"
        self.actions = actions
        self.gamma = gamma
        self.alpha_decay_rate = alpha_decay_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = str(observation)
        self.previous_action_id = None
        self.q_values = self._init_q_values()
        self.is_share = False

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

    def init_policy(self, policy):
        self.policy = policy

    def act(self, q_values=None, step=0):
        action_id = self.policy.select_action(self.q_values[self.state])
        self.previous_action_id = action_id
        action = self.actions[action_id]
        return action

    def observe_state_and_reward(self, next_state, reward):
        """
            次の状態と報酬の観測 
        """
        self.observe(next_state)
        self.get_reward(reward)

    def observe(self, next_state):
        next_state = str(next_state)
        if next_state not in self.q_values: # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

    def get_reward(self, reward, is_finish=True, step=0):
        """
            報酬の獲得とQ値の更新 
        """
        self.reward_history.append(reward)
        self.q_values[self.previous_state][self.previous_action_id] = self._update_q_value(reward)

    def _update_q_value(self, reward):
        """
            Q値の更新 
        """
        q = self.q_values[self.previous_state][self.previous_action_id] # Q(s, a)
        max_q = max(self.q_values[self.state]) # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        updated_q = q + (self.alpha * (reward + (self.gamma*max_q) - q))
        return updated_q
