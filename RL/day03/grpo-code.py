import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self.pi = PolicyNet()  # 策略: pi_theta, π_θ
        self.lr_pi = 0.002
        self.optimizer_pi = torch.optim.Adam(
            self.pi.parameters(), lr=self.lr_pi)

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

        # [logπ_θ_old(A_0|S_0), logπ_θ_old(A_1|S_1), ..., logπ_θ_old(A_T|S_T)]
        old_log_probs = []

        done = False  # 初始化为游戏没结束

        while not done:
            action, probs = self.get_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 在环境中执行动作

            states.append(state)  # S_t
            actions.append(action)  # A_t
            rewards.append(reward)  # R_t
            # `torch.log(probs[action]).item()`: logπ_θ_old(A_t|S_t)
            old_log_probs.append(torch.log(probs[action]).item())

            # 状态转移
            state = next_state

        states = torch.tensor(states)
        actions = torch.tensor(actions).view(-1, 1)
        R = sum(rewards)  # 一条轨迹的总的奖励
        old_log_probs = torch.tensor(old_log_probs).view(-1, 1)

        return states, actions, R, old_log_probs

    def calculate_grpo_advantages(self, trajectories):
        """计算一组轨迹中每条轨迹的组内优势"""
        # [reward_0, reward_1, ... reward_{G-1}]
        R_s = [R for (_, _, R, _) in trajectories]  # 将每条轨迹的总奖励提取出来
        # 一组轨迹回报的均值
        r_mean = np.mean(R_s)
        # 一组轨迹回报的标准差
        r_std = np.std(R_s) + 1e-8
        # 计算一组轨迹中每条轨迹的组内优势
        rewards_in_group = [(R - r_mean) / r_std for R in R_s]
        return rewards_in_group

    def update(self, trajectories):
        rewards_in_group = self.calculate_grpo_advantages(trajectories)
        for _ in range(20):  # 使用一组轨迹更新20次策略模型
            obj = 0.0
            for tau, r_in_group in zip(trajectories, rewards_in_group):
                states, actions, R, old_log_probs = tau
                log_probs = torch.log(self.pi(states).gather(1, actions))
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio,  1 - 0.2, 1 + 0.2)
                obj += torch.mean(torch.min(ratio * r_in_group,
                                            clipped_ratio * r_in_group))
            obj = obj / len(rewards_in_group)

            loss_pi = -obj
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()


# 创建推车环境
env = gym.make("CartPole-v0")
agent = Agent()

returns = []
episodes = []
G = 8  # 一组轨迹8条

for step in range(1, 100 + 1):
    # ① 采样一组轨迹轨迹
    trajectories = []
    for _ in range(G):
        trajectory = agent.rollout(env)
        trajectories.append(trajectory)
    # ② 使用一组轨迹更新策略
    agent.update(trajectories)
    
    r_mean = sum(R for _, _, R, _ in trajectories) / len(trajectories)
    returns.append(r_mean)
    episodes.append(step)

    if step % 10 == 0:
        print(f"Step: {step}, 一组轨迹回报的均值为：{r_mean}")


def plot_loss(episodes, returns, filename):
    f = plt.figure()
    plt.plot(episodes, returns)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    f.savefig(filename, bbox_inches="tight")
    plt.show()


plot_loss(episodes, returns, "pg-loss.pdf")
