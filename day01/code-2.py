import gym
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn


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
        self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), lr=self.lr_pi)

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

    def update(self, trajectory):
        states, actions, rewards = trajectory

        # G(τ)
        G = 0.0
        for r in rewards[::-1]:
            G = r + self.gamma * G

        states = torch.tensor(states)  # [S_0, S_1, ..., S_T], (B, 4)
        # [A_0, A_1, ..., A_T]
        #       |
        #       v
        # [[A_0], [A_1], ..., [A_T]]
        # (B,) --> (B, 1)
        actions = torch.tensor(actions).view(-1, 1)
        log_action_probs = torch.log(self.pi(states).gather(1, actions))

        obj = torch.sum(log_action_probs) * G

        loss = -obj

        self.optimizer_pi.zero_grad()
        loss.backward()
        self.optimizer_pi.step()


# 创建推车环境
env = gym.make("CartPole-v0")
agent = Agent()

returns = []
episodes = []

for step in range(1, 3000 + 1):
    # ① 采样一条轨迹
    trajectory = agent.rollout(env)
    # ② 更新策略
    agent.update(trajectory)

    returns.append(sum(trajectory[2]))
    episodes.append(step)
    if step % 100 == 0:
        print(f"Step: {step}, 奖励总和：{sum(trajectory[2])}")


def plot_loss(episodes, returns, filename):
    f = plt.figure()
    plt.plot(episodes, returns)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    f.savefig(filename, bbox_inches="tight")
    plt.show()


plot_loss(episodes, returns, "pg-loss.pdf")
