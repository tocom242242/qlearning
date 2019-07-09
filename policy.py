import copy
import numpy as np

class EpsGreedyQPolicy():
    """
        ε-greedy選択 
    """
    def __init__(self, epsilon=.1, decay_rate=1):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.name = "EPS"

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.random_integers(0, nb_actions-1)
        else:   # greedy 行動
            action = np.argmax(q_values)

        return action
