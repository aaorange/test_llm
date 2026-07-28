import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """策略网络：π_θ"""

    def __init__(self):
        super().__init__()
        # 状态：4个浮点数
        # 动作的概率分布：二项分布
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 2)

    def forward(self, x):
        """
        x: 一批状态，形状：(B, 4)
        输出的形状：(B, 2)
        """
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98  # 折扣因子: γ
        self.pi = PolicyNet()  # 策略: pi_theta, π_θ
        self.lr_pi = 0.002
        self.optim = torch.optim.Adam(self.pi.parameters(), lr=self.lr_pi)

    def get_action(self, state):
        """state: (4,)"""
        # (4,) --> (1,4) --> (1,2) ---> (2,)
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        # 创建一个二项分布采样器，根据概率分布probs采样一个动作
        action = torch.multinomial(probs, num_samples=1).item()
        return action, probs

    def rollout(self, env):
        """在环境env中采样一条轨迹trajectory"""
        state = env.reset()  # S_0
        states = []  # [S_0, S_1, ..., S_T]
        actions = []  # [A_0, A_1, ..., A_T]
        rewards = []  # [R_0, R_1, ..., R_T]

        done = False  # 初始化为游戏没结束

        while not done:
            action, _ = self.get_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 在环境中执行动作

            states.append(state)  # S_t
            actions.append(action)  # A_t
            rewards.append(reward)  # R_t

            # 状态转移
            state = next_state

        return states, actions, rewards


# 创建推车环境
env = gym.make("CartPole-v0")
agent = Agent()
tau = agent.rollout(env)

print(tau)

# 如何计算轨迹τ的带折扣因子的回报？
G = 0.0
rewards = tau[2]

for r in rewards[::-1]:
    G = r + agent.gamma * G

print(G)
