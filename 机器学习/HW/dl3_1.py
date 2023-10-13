import gym
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import pygame
import time
import randomimport gym
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import pygame
import time
import random

# 主要函数体和类的编写
class QLearningAgent():
    def __init__(self, obs_n, act_n, lr, gamma, e_greed):
        self.act_n = act_n  # 动作数
        self.lr = lr  # 学习率
        self.gamma = gamma  # 奖励衰减
        self.epsilon = e_greed  # 贪心率
        self.Q = np.zeros((obs_n, act_n))  # 存放Q值

    # 根据输入观察值根据贪心策略输出动作之
    def sample(self, obs):
        # 自己编写
        rand = random.random()
        if rand < self.epsilon:
            action = random.randint(0, self.act_n - 1)
        else:
            action = self.predict(obs)
        return action

    # 根据观测值输出当前Q表下对应的最优动作
    def predict(self, obs):
        # 自己编写
        QList = self.Q[obs, :]
        Qmax = np.max(QList)
        actionList = np.where(QList == Qmax)[0]
        action = np.random.choice(actionList)
        return action

    # Q表更新学习
    def learn(self, obs, action, reward, next_obs, done):
        # 自己编写
        Q_predict = self.Q[obs, action]
        if done is not True:
            Q_target = reward + self.gamma * np.max(self.Q[next_obs, :])
        else:
            Q_target = reward
        self.Q[obs, action] += self.lr * (Q_target - Q_predict)

# 训练函数
def train_episode_Q(env, agent):
    # 自己编写
    train_steps = 0
    train_reward = 0
    obs, _ = env.reset()
    while True:
        action = agent.sample(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.learn(obs, action, reward, next_obs, done)
        obs = next_obs
        train_reward += reward
        train_steps += 1
        if done:
            break
    return train_reward, train_steps


# 测试函数
def test_episode_Q(env, agent):
    # 自己编写
    test_reward = 0
    obs, _ = env.reset()
    figure = plt.imshow(env.render())
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _, _ = env.step(action)
        test_reward += reward
        obs = next_obs
        time.sleep(0.5)
        # 可视化
        data=env.render()
        figure.set_data(data)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        
        if done:
            break
    return test_reward

# 主程序编写
# 1.两种策略下训练过程的Reward变化曲线可视化，使用matplotlib实现
#2.两种策略得到的最优路径对比：参考说明文档中的动画演示代码实现最优路径的演示
env = gym.make("CliffWalking-v0",render_mode='rgb_array')  # 0 up, 1 right, 2 down, 3 left
env.reset()

agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        lr=0.1,
        gamma=0.9,
        e_greed=0.1)

train_ = []
for episode in range(500):
        ep_reward, ep_steps = train_episode_Q(env, agent)
        # print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))
        train_.append(ep_reward)

plt.figure()
plt.plot(range(500), train_)
plt.show()


test_reward = test_episode_Q(env, agent)
print('test reward = %.1f' % test_reward)


# 仿照QlearningAgent完成SarsaAgent的编写
class SarsaAgent():
    def __init__(self, obs_n, act_n, lr, gamma, e_greed):
        self.act_n = act_n  # 动作数
        self.lr = lr  # 学习率
        self.gamma = gamma  # 奖励衰减
        self.epsilon = e_greed  # 贪心率
        self.Q = np.zeros((obs_n, act_n))  # 存放Q值

    # 根据输入观察值根据贪心策略输出动作之
    def sample(self, obs):
        # 自己编写
        rand = random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.act_n)
        else:
            action = self.predict(obs)
        return action

    # 根据观测值输出当前Q表下对应的最优动作
    def predict(self, obs):
        # 自己编写
        QList = self.Q[obs, :]
        Qmax = np.max(QList)
        actionList = np.where(QList == Qmax)[0]
        action = np.random.choice(actionList)
        return action

    # Q表更新学习
    def learn(self, obs, action, reward, next_obs, next_action, done):
        # 自己编写
        Q_predict = self.Q[obs, action]
        if done is not True:
            Q_target = reward + self.gamma * self.Q[next_obs, next_action]
        else:
            Q_target = reward
        self.Q[obs, action] += self.lr * (Q_target - Q_predict)
        
# 训练函数
def train_episode_S(env, agent):
    # 自己编写
    train_steps = 0
    train_reward = 0
    obs, _ = env.reset()
    action = agent.sample(obs) 
    while True:
        next_obs, reward, done, _, _ = env.step(action)
        next_action = agent.sample(next_obs)
        agent.learn(obs, action, reward, next_obs, next_action, done)
        action = next_action
        obs = next_obs
        train_reward += reward
        train_steps += 1 
        if done:
            break
    return train_reward, train_steps


# 测试函数
def test_episode_S(env, agent):
    # 自己编写
    test_reward = 0
    obs, _ = env.reset()
    figure = plt.imshow(env.render())
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _, _ = env.step(action)
        test_reward += reward
        obs = next_obs
        time.sleep(0.5)
        # 可视化
        data=env.render()
        figure.set_data(data)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        
        if done:
            break
    return test_reward

# 主程序编写
# 1.两种策略下训练过程的Reward变化曲线可视化，使用matplotlib实现
#2.两种策略得到的最优路径对比：参考说明文档中的动画演示代码实现最优路径的演示
env = gym.make("CliffWalking-v0",render_mode='rgb_array')  # 0 up, 1 right, 2 down, 3 left
env.reset()

agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        lr=0.1,
        gamma=0.9,
        e_greed=0.1)

train_ = []
for episode in range(500):
        ep_reward, ep_steps = train_episode_S(env, agent)
        # print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))
        train_.append(ep_reward)

plt.figure()
plt.plot(range(500), train_)
plt.show()


test_reward = test_episode_S(env, agent)
print('test reward = %.1f' % test_reward)


#3.不同贪心值的探究实验，从前两个任务的角度综合分析

env = gym.make("CliffWalking-v0",render_mode='rgb_array')
env.reset()


for e_ in (0.01, 0.05, 0.1):
    agent_Q = QLearningAgent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    lr=0.1,
    gamma=0.9,
    e_greed=0.1)
    
    agent_S = SarsaAgent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    lr=0.1,
    gamma=0.9,
    e_greed=0.1)





