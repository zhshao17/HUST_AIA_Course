import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
from torch import optim

class DQN(nn.Module):
    def __init__(self, n_actions, n_inputs, lr=0.01):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 64)  # 输入维度为4
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, n_actions)  # 输出维度为2，某个状态下每个动作空间的Q值

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = 'cpu'
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent:
    def __init__(self, env, n_actions, state_n_dim, gamma=0.99, epsilon=1):
        self.env = env
        self.n_actions = n_actions
        self.state_n_dim = state_n_dim
        self.gamma = gamma  # 未来值影响参数
        self.epsilon = epsilon  # 探索策略
        self.eps_min = 0.05  # epsilon会衰减，但不低于eps_min
        self.eps_dec = 1e-4  # 每次衰减0.0005
        self.iter_count = 0 #记录学习次数
        self.policy_net = DQN(self.n_actions, self.state_n_dim)    #待学习的网络
        self.target_net = DQN(self.n_actions, self.state_n_dim)  # 目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())#初始目标网络参数同待学习的网络
        self.replay_memory = collections.deque(maxlen=10000)#经验池
        self.min_replay_memory_size = 100  # 经验池至少要100个transition
        self.batch_size = 64    #每次从经验池中取64个trainsition学习
        self.update_target = 10  # 每学习10次更新一次target网络
        self.scores = []#记录每个episode的得分
    def update_replay_memory(self, obs):
        self.replay_memory.append(obs)

    def choose_action(self, state, score):
        action_selected = set(np.where(state == 1)[0])  # 已选择了的动作集合
        if np.random.random() > self.epsilon and score >= 0:
            q = self.policy_net(torch.tensor([state], dtype=torch.float32))
            print('处理前的q: ', q) # tensor([[-0.2269, -0.0726, -0.1312,  0.0524,  0.0142]],grad_fn=<AddmmBackward0>)
            # qq = q.detach().numpy()[0] # 不能修改tensor???
            # # print('处理前的qq: ', qq)
            # for index in action_selected:
            #     qq[index] = -10000.00
            # # print('处理后的qq: ', qq)
            # action = qq.argmax()  # 值最大的下标
            action = torch.argmax(q).item()
            print('根据网络输出，选择最大值对应的下标，作为action:', action)
        else:
            action_list = list(set(self.env.action_space).difference(action_selected)) # 可选动作
            action = np.random.choice(action_list) # 随机采样一个动作
            print('随机采样得到action:', action)
        return action

    def train(self):
        len_replay_memory = len(self.replay_memory)
        if len_replay_memory < self.batch_size:
            print('经验池内的transition不够，目前:', len_replay_memory, '条！')
            return
        self.policy_net.optimizer.zero_grad()#清空gradient buffer
        batch = random.sample(self.replay_memory, self.batch_size)  # 随机采样结果[(state, action, reward, next_state, done),(),..,()]
        states, actions, rewards, next_states, dones = [trans[0] for trans in batch], [trans[1] for trans in batch], [
            trans[2] for trans in batch], [trans[3] for trans in batch], [trans[4] for trans in batch]
        state_batch = torch.tensor(states, dtype=torch.float32) # tensor([0,0,0,0,0],[1,0,0,0,0])
        # print(state_batch)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32)
        # print(next_state_batch)
        action_batch = torch.tensor(actions).numpy()  # tensor([0, 1, 2, 1, 4,...],dtype=torch.int32)
        # print(action_batch) # [1 4 0 2 4 0 1 2 4 3 1 3 3 3...]
        reward_batch = torch.tensor(rewards)
        # print(reward_batch)
        done_batch = torch.tensor(dones)  # tensor([False, False,False,...,False,False,True,False,False,True,...])
        batch_index = np.arange(self.batch_size, dtype=np.int32)  # [0 1 2 3 ...]

        pred_list = self.policy_net(state_batch)[batch_index, action_batch]  # 预测值tensor([0.89,0.98,0.7,0.9,0.1])
        next_action_list = self.target_net(next_state_batch)  # [[0.89,0.98,0.7,0.9,0.1],[],...]
        next_action_list[done_batch] = 0.0  # 下一步结束，y = r;其他,y = r + gamma * maxQ(next_s,a)-->[[0.02,0,89],[0.98,0.01],[0.0,0.0],...]
        new_q = reward_batch + self.gamma * torch.max(next_action_list, dim=1)[0]  # 取下一个状态动作对应价值最大的

        loss = self.policy_net.loss(pred_list, torch.as_tensor(new_q, dtype=torch.float32))  # as_tensor()修改数组值，张量值也会变
        loss.backward()  # 反向计算梯度
        self.policy_net.optimizer.step()  # 梯度传播，更新参数
        self.iter_count += 1
        if not self.iter_count % self.update_target:  # 每学习10次更新一次target网络
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def step(self):  # 完成一个episode
        done = False
        state = self.env.reset()  # [0 0 0 0 0]
        print('游戏开始，得到一个状态state:', state)
        episode_reward = 0
        reward = 0
        while not done:
            action = self.choose_action(state,reward)  # 返回最大值对应的下标,作为action,如 0
            next_state, reward, done, repet = self.env.step(action) # [1,0,0,0,0] 1 False
            print('环境交互得到next_s:', next_state, 'reward: ', reward, 'done: ', done)
            if reward > 0:
                episode_reward = reward
            # if repet == 0: #repet=1时是错误的transition，不应该学习
            self.update_replay_memory((state, action, reward, next_state, done))  # trainsition放入经验池
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
            print('完成一个step，epsilon更新为：', self.epsilon)
            state = next_state
        print('Episode结束， reward为: ', episode_reward)
        self.scores.append(episode_reward)
    def save_model(self):
        torch.save(self.policy_net.state_dict(), 'policy_net.pth')  # 保存模型参数

if __name__ == '__main__':
    round_count = 3  # 跑5次的结果取平均
    round_all_score = 0
    env = gym.make('MyEnv-v0')
    n_actions = env.n_actions  # 可选动作有5个
    state_n_dims = env.state_dim  # 状态的维度为5
    for i in range(round_count):
        agent = Agent(env, n_actions, state_n_dims)
        episodes = 2000  # 每次跑900个回合
        for episode in range(episodes):
            agent.step()  # 完成一个episode，将每个step的trainsition放入经验池
            agent.train()  # 利用经验池中的trainsition学习
            print('Episode: ', episode, '| reward: ', agent.scores[episode])
        avg_score = np.mean(agent.scores)  # 900个episodes的平均分
        print('Round: ', i, '| Average score: ', int(avg_score))
        round_all_score += avg_score
        agent.env.close()
    fig = plt.figure(figsize=(7, 7))  # figsize是图片的大小`
    plt.plot(range(episodes),agent.scores,'g-')
    plt.xlabel(u'iters')
    plt.ylabel(u'scores')
    plt.show()
    print('run ', round_count, 'rounds,the score is: ', int(round_all_score / round_count))

import gym
import numpy as np
from numpy import random
import time

class MyEnv(gym.Env):
    def __init__(self):
        self.viewer = None
        # 状态空间和回报
        # self.files = [(1,1),(2,6),(5,18),(6,22),(7,28)] # 定义文件的大小和回报
        self.files = [(2,3),(3,4),(4,5),(5,6),(4,3),(7,12),(3,3),(2,2)]
        self.min_weight = 2
        self.limit_weight = 8 #网关最大容量
        self.state_dim = len(self.files) # 状态文件个数
        # 动作空间
        self.action_space = np.arange(len(self.files)) # [0,1,2,3,4]
        self.n_actions = self.state_dim # 定义可选动作个数
        self.repet = 0 # 标志是否重复放入
    def step(self, action):
        if self.state[action] == 1: # 接下来要选的动作已选中
            weight_sum = self.state[action] * self.files[action][0]
            for i in range(self.state_dim):
                weight_sum += self.state[i] * self.files[i][0]
            if weight_sum > self.limit_weight: # 超重
                is_terminal = True
                r = -30
                # self.repet = 1 # 标志这条transition不能用于学习
            else:
                is_terminal = False
                r = -30
            next_state = self.state
        else:
            #系统当前状态
            state = self.state
            state[action] = 1 #[1,0,0,0,0]
            weight_sum = 0
            for i in range(self.state_dim):
                weight_sum += self.state[i] * self.files[i][0]
            if weight_sum > self.limit_weight: # 超重
                is_terminal = True
                r = -30
            else:
                is_terminal = False
                r = 0
                for i in range(self.state_dim):
                    r += self.state[i] * self.files[i][1]
            next_state = state
            self.state = next_state
        if weight_sum + self.min_weight > self.limit_weight:
            is_terminal = True

        return next_state, r, is_terminal, self.repet

    def reset(self):
        self.state = np.array([0,0,0,0,0,0,0,0]) # 网关为空
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering
        width = 60
        height = 40
        edge_x = 0
        edge_y = 0
        if self.viewer is None:
            self.viewer = rendering.Viewer(300, 200)

        # 右下角                 用黑色表示墙
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 2, edge_y + height * 1)))
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 3, edge_y + height * 1)))
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 4, edge_y + height * 1)))
        # 左边
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(rendering.Transform((edge_x, edge_y + height * 3)))
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 1, edge_y + height * 3)))
        # 上边
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 3, edge_y + height * 4)))
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 3, edge_y + height * 5)))
        # 出口，用黄色表示出口
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(1, 0.9, 0)).add_attr(
            rendering.Transform((edge_x + width * 4, edge_y + height * 3)))
        # 画网格
        for i in range(1, 7):
            self.viewer.draw_line((edge_x, edge_y + height * i), (edge_x + 5 * width, edge_y + height * i))  # 横线
            self.viewer.draw_line((edge_x + width * (i - 1), edge_y + height),
                                  (edge_x + width * (i - 1), edge_y + height * 6))  # 竖线

        # 人的像素位置
        self.x = [edge_x + width * 0.5, edge_x + width * 1.5, edge_x + width * 2.5, 0, edge_x + width * 4.5,
                  edge_x + width * 0.5, edge_x + width * 1.5, edge_x + width * 2.5, 0, edge_x + width * 4.5,
                  0, 0, edge_x + width * 2.5, edge_x + width * 3.5, edge_x + width * 4.5,
                  edge_x + width * 0.5, edge_x + width * 1.5, edge_x + width * 2.5, edge_x + width * 3.5,
                  edge_x + width * 4.5,
                  edge_x + width * 0.5, edge_x + width * 1.5, 0, 0, 0]

        self.y = [edge_y + height * 5.5, edge_y + height * 5.5, edge_y + height * 5.5, 0, edge_y + height * 5.5,
                  edge_y + height * 4.5, edge_y + height * 4.5, edge_y + height * 4.5, 0, edge_y + height * 4.5,
                  0, 0, edge_y + height * 3.5, edge_y + height * 3.5, edge_y + height * 3.5,
                  edge_y + height * 2.5, edge_y + height * 2.5, edge_y + height * 2.5, edge_y + height * 2.5,
                  edge_y + height * 2.5,
                  edge_y + height * 1.5, edge_y + height * 1.5, 0, 0, 0]
        # 用圆表示人
        # self.viewer.draw_circle(18,color=(0.8,0.6,0.4)).add_attr(rendering.Transform(translation=(edge_x+width/2,edge_y+height*1.5)))
        self.viewer.draw_circle(18, color=(0.8, 0.6, 0.4)).add_attr(
            rendering.Transform(translation=(self.x[self.state - 1], self.y[self.state - 1])))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

# env=gym.make('MyEnv-v0')
# state = env.reset()
# print('初始状态state: ', state)
# n_act = env.action_space
# print('可选动作个数：',n_act)
# reward=0
# while True:
#         action = env.actions[int(random.random()*len(env.actions))]
#         next_state,r,is_terminal,info = env.step(action)
#         env.render()
#         reward += r
#         if is_terminal == True:
#             print("游戏结束，reward:",reward)
#             time.sleep(18)
#             break
#         time.sleep(1)

