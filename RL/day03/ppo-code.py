import matplotlib.pyplot as plt
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


class ValueNet(nn.Module):
    """价值函数网络：V_ω"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 4)
        # 输出：(B, 1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 1.0  # 折扣因子: γ
        self.lmbda = 0.95  # 广义优势估计的超参数：λ
        self.pi = PolicyNet()  # 策略: pi_theta, π_θ
        self.v = ValueNet()
        self.lr_pi = 0.002
        self.lr_v = 0.05
        self.optimizer_pi = torch.optim.Adam(
            self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = torch.optim.Adam(self.v.parameters(), lr=self.lr_v)

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
        next_states = []  # [S_1, S_2, ..., S_{T+1}]
        actions = []  # [A_0, A_1, ..., A_T]
        rewards = []  # [R_0, R_1, ..., R_T]
        dones = []  # [False, False, ..., True]

        # [logπ_θ_old(A_0|S_0), logπ_θ_old(A_1|S_1), ..., logπ_θ_old(A_T|S_T)]
        old_log_probs = []

        done = False  # 初始化为游戏没结束

        while not done:
            action, probs = self.get_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 在环境中执行动作

            states.append(state)  # S_t
            next_states.append(next_state)
            actions.append(action)  # A_t
            rewards.append(reward)  # R_t
            dones.append(done)
            # `torch.log(probs[action]).item()`: logπ_θ_old(A_t|S_t)
            old_log_probs.append(torch.log(probs[action]).item())

            # 状态转移
            state = next_state

        states = torch.tensor(states)
        next_states = torch.tensor(next_states)
        actions = torch.tensor(actions).view(-1, 1)
        rewards = torch.tensor(rewards).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1)
        old_log_probs = torch.tensor(old_log_probs).view(-1, 1)

        # 单步TD目标
        # [R_0 + γV(S_1), R_1 + γV(S_2), ..., R_T]
        td_targets = rewards + self.gamma * self.v(next_states) * (1 - dones)
        # 单步TD目标作为预测目标，是常量
        td_targets = td_targets.detach()
        # 单步TD误差
        # [R_0 + γV(S_1) - V(S_0), R_1 + γV(S_2) - V(S_1), ..., R_T - V(S_T)]
        td_errors = td_targets - self.v(states)
        # 单步TD误差在计算策略梯度的时候，是常量, (B, 1) --> (B,)
        td_errors = td_errors.squeeze(1).detach().numpy().tolist()

        # 计算广义优势估计
        gae_list_reversed = []
        lastgae = 0.0
        # 逆序遍历单步TD误差
        # GAE_{t} = δ_t + γλGAE_{t+1}
        for delta in td_errors[::-1]:
            lastgae = delta + self.gamma * self.lmbda * lastgae
            gae_list_reversed.append(lastgae)

        gae_list = list(reversed(gae_list_reversed))  # (B,)
        gae_list = torch.tensor(gae_list).view(-1, 1)  # (B, 1)
        # 广义优势目标 = GAE + V(S_t)
        # 广义优势目标是V_ω(S_t)的逼近目标
        gae_targets = gae_list + self.v(states)
        gae_targets = gae_targets.detach()

        return states, actions, rewards, gae_targets, gae_list, old_log_probs

    def update(self, trajectory):
        """实现的是PPO伪代码中的内层循环"""
        states, actions, rewards, gae_targets, gae_list, old_log_probs = trajectory

        # ppo伪代码的内层循环，一条轨迹使用10次
        for _ in range(10):
            # [logπ_θ(A_0|S_0), logπ_θ(A_1|S_1), ..., logπ_θ(A_T|S_T)]
            log_probs = torch.log(self.pi(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
            # ① 策略函数的目标函数
            obj = torch.sum(torch.min(ratio * gae_list,
                            clipped_ratio * gae_list))
            loss_pi = - obj

            # ② 价值网络的损失函数MSE
            loss_v = F.mse_loss(self.v(states), gae_targets)

            self.optimizer_pi.zero_grad()
            self.optimizer_v.zero_grad()
            loss_pi.backward()
            loss_v.backward()
            self.optimizer_pi.step()
            self.optimizer_v.step()


# 创建推车环境
env = gym.make("CartPole-v0")
agent = Agent()

returns = []
episodes = []

for step in range(1, 500 + 1):
    # ① 采样一条轨迹
    trajectory = agent.rollout(env)
    # ② 更新策略
    agent.update(trajectory)

    returns.append(sum(trajectory[2]).item())
    episodes.append(step)
    if step % 100 == 0:
        print(
            f"Step: {step}, 奖励总和：{sum(trajectory[2]).item()}, V(S_0): {(trajectory[3] - trajectory[4])[0].item()}")


def plot_loss(episodes, returns, filename):
    f = plt.figure()
    plt.plot(episodes, returns)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    f.savefig(filename, bbox_inches="tight")
    plt.show()


plot_loss(episodes, returns, "pg-loss.pdf")
