import os,sys
import random
import numpy as np
import matplotlib.pyplot as plt
from agents.qlearning_agent import QLearningAgent
from agents.policy import EpsGreedyQPolicy
from envs.grid_world import GridWorld

if __name__ == '__main__':
    grid_env = GridWorld() # grid worldの環境の初期化
    ini_state = grid_env.start_pos  # 初期状態（エージェントのスタート地点の位置）
    policy = EpsGreedyQPolicy(epsilon=0.01) # 方策の初期化。ここではε-greedy
    agent = QLearningAgent(actions=np.arange(4), observation=ini_state, policy=policy) # Q Learning エージェントの初期化
    nb_episode = 100   #エピソード数
    rewards = []    # 評価用報酬の保存
    is_goal = False # エージェントがゴールしてるかどうか？
    for episode in range(nb_episode):
        episode_reward = [] # 1エピソードの累積報酬
        while(is_goal == False):    # ゴールするまで続ける
            action = agent.act()    # 行動選択
            state, reward, is_goal = grid_env.step(action)
            agent.observe_state_and_reward(state, reward)   # 状態と報酬の観測
            episode_reward.append(reward)
        rewards.append(np.sum(episode_reward)) # このエピソードの平均報酬を与える
        state = grid_env.reset()    #  初期化
        agent.observe(state)    # エージェントを初期位置に
        is_goal = False

    # 結果のプロット
    plt.plot(np.arange(nb_episode), rewards)
    plt.xlabel("episode")
    plt.ylabel("accumulated reward")
    plt.show()
    plt.savefig("result.jpg")
