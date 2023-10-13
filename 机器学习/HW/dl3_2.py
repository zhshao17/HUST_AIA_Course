import torch
import torch.nn as nn
import numpy as np
import gym
import torch.autograd
import random
print(gym.__version__)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 24),  ##两个输入
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 3)  ##三个输出
        )
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.fc(x)


env = gym.envs.make('MountainCar-v0')
env = env.unwrapped
net = MyNet()  # 实例化
net2 = MyNet()

store_count = 0
store_size = 2000
decline = 0.6  # epsilo
learn_time = 0
updata_time = 20  # 目标值网络更新步长
gama = 0.9
b_size = 1000
store = np.zeros((store_size, 6))  ###[s,a,s_,r]，其中s占两个，a占一个，r占一个
start_study = False

for i in range(50000):
    s = env.reset()  ##
    while True:
        ###根据 state 产生动作
        if random.randint(0, 100) < 100 * (decline ** learn_time):  # 相当于epsilon
            a = random.randint(0, 2)
        else:
            out = net(torch.Tensor(s)).detach()  ##detch()截断反向传播的梯度，[r1,r2]
            a = torch.argmax(out).data.item()  ##[取最大，即取最大值的index]
        s_, r, done, info = env.step(a)  ##环境返回值，可查看step函数
        r = s_[0] + 0.5
        if s_[0] > -0.5:
            r = s_[0] + 0.5
            if s_[0] > 0.5:
                r = 5
        else:
            r = 0
        # r = abs(s_[0]-(-0.5))

        store[store_count % store_size][0:2] = s  ##覆盖老记忆
        store[store_count % store_size][2:3] = a
        store[store_count % store_size][3:5] = s_
        store[store_count % store_size][5:6] = r
        store_count += 1
        s = s_
        #####反复试验然后存储数据，存满后，就每次取随机部分采用sgd
        if store_count > store_size:
            if learn_time % updata_time == 0:
                net2.load_state_dict(net.state_dict())  ##延迟更新

            index = random.randint(0, store_size - b_size - 1)
            b_s = torch.Tensor(store[index:index + b_size, 0:2])
            b_a = torch.Tensor(store[index:index + b_size, 2:3]).long()  ##  因为gather的原因，索引值必须是longTensor
            b_s_ = torch.Tensor(store[index:index + b_size, 3:5])
            b_r = torch.Tensor(store[index:index + b_size, 5:6])  # 取batch数据

            q = net(b_s).gather(1, b_a)  #### 聚合形成一张q表    根据动作得到的预期奖励是多少
            q_next = net2(b_s_).detach().max(1)[0].reshape(b_size, 1)  # 值和索引，延迟更新
            tq = b_r + gama * q_next
            loss = net.mls(q, tq)
            net.opt.zero_grad()
            loss.backward()
            net.opt.step()

            learn_time += 1
            if not start_study:
                print('start_study')
                start_study = True
                break
        if done:
            print(i)
            break

        env.render()
