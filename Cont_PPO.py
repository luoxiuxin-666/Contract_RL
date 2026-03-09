import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # 需要用到 F.softplus
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class WeightedActorCritic(nn.Module):
    def __init__(self, state_dim, n_dn, n_cn, action_std_init=0.6):
        super(WeightedActorCritic, self).__init__()
        self.n_dn = n_dn
        self.n_cn = n_cn

        # =====================================================
        # 动作维度定义
        # =====================================================

        # 1. 数据合同 (DN):
        # 为了生成 n_dn 个单调递增的值，我们需要：
        # - n_dn 个 delta 值 (控制形状)
        # - 1 个 scale 值 (控制整体大小)
        # 所以 Raw Action 维度 = n_dn + 1
        self.dim_dn_raw = n_dn + 1
        self.dim_dn_out = n_dn  # 实际输出给环境的维度

        # 2. 算力合同 (CN)
        self.dim_cn = n_cn

        # 3. 带宽分配 (BW)
        self.dim_bw = n_dn + n_cn

        # 4. 算力节点权重 (CN Weights)
        self.dim_weights = n_cn

        # 总 Raw Action 维度 (神经网络输出层大小)
        self.action_dim = self.dim_dn_raw + self.dim_cn + self.dim_bw + self.dim_weights

        # =====================================================
        # 网络结构
        # =====================================================
        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # Actor Body
        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh()
        )

        # Actor Head
        self.actor = layer_init(nn.Linear(64, self.action_dim), std=0.01)
        self.action_var = nn.Parameter(torch.full((self.action_dim,), action_std_init * action_std_init))

    def act(self, state):
        features = self.actor_body(state)
        action_mean = self.actor(features)

        # 协方差矩阵
        action_std = torch.nn.functional.softplus(self.action_var) + 1e-5
        action_var_exp = action_std.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var_exp).to(device)

        # 采样 Raw Action (高斯分布)
        dist = MultivariateNormal(action_mean, cov_mat)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action)

        # =====================================================
        # 物理映射 (核心修改部分)
        # =====================================================

        # --- 切片索引计算 ---
        idx_1 = self.dim_dn_raw  # DN 结束位置 (N+1)
        idx_2 = idx_1 + self.dim_cn
        idx_3 = idx_2 + self.dim_bw

        # --- A. 数据合同 (DN): 差分 + 缩放 -> 单调递增 ---
        dn_raw_part = raw_action[..., :idx_1]

        # A1. 提取 Deltas (前 N 位) -> 必须 > 0 -> Softplus
        # 对应 raw_action 的 [0, 1, ..., N-1]
        deltas = F.softplus(dn_raw_part[..., :-1])

        # A2. 提取 Scale (最后 1 位) -> (0, 1) -> Sigmoid
        # 对应 raw_action 的 [N]
        scale = torch.sigmoid(dn_raw_part[..., -1:])

        # A3. 生成形状 (Cumsum) -> 单调递增
        shape_curve = torch.cumsum(deltas, dim=-1)

        # A4. 归一化并缩放 (重要！)
        # 如果不归一化，deltas 的累加值可能非常大，scale 就失效了。
        # 我们把曲线最大值归一化到 1，然后乘以 scale。
        # epsilon 防止除以 0
        max_val = shape_curve[..., -1:].clone()
        max_val[max_val < 1e-6] = 1.0

        # 结果范围: [0, scale]
        dn_proc = (shape_curve / max_val) * scale

        # --- B. 算力合同 (CN) -> Sigmoid ---
        cn_proc = torch.sigmoid(raw_action[..., idx_1:idx_2])

        # --- C. 带宽分配 (BW) -> Softmax ---
        bw_proc = torch.softmax(raw_action[..., idx_2:idx_3], dim=-1)

        # --- D. 算力权重 (Weights) -> Sigmoid ---
        weights_proc = torch.sigmoid(raw_action[..., idx_3:])

        # 拼接最终物理动作
        processed_action = torch.cat([dn_proc, cn_proc, bw_proc, weights_proc], dim=-1)

        return processed_action.detach(), raw_action.detach(), log_prob.detach(), self.critic(state).detach()

    def evaluate(self, state, raw_action):
        # Evaluate 保持不变，因为它计算的是 Raw Action 的概率分布
        # PPO 优化的是产生这个 Raw Action 的概率
        features = self.actor_body(state)
        action_mean = self.actor(features)

        action_std = torch.nn.functional.softplus(self.action_var) + 1e-5
        action_var_exp = action_std.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var_exp).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        log_prob = dist.log_prob(raw_action)
        dist_entropy = dist.entropy()

        return log_prob, self.critic(state), dist_entropy


class PPO:
    def __init__(self, state_dim, n_dn, n_cn, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.buffer = RolloutBuffer()
        # 注意：这里传入 n_dn，内部会自动处理为 n_dn + 1
        self.policy = WeightedActorCritic(state_dim, n_dn, n_cn).to(device)
        self.MseLoss = nn.MSELoss()

        self.optimizer = optim.Adam([
            {'params': self.policy.actor_body.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.action_var, 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = WeightedActorCritic(state_dim, n_dn, n_cn).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if state.dim() == 1: state = state.unsqueeze(0)

            # 返回处理后的动作(用于环境) 和 原始动作(用于存储训练)
            proc_action, raw_action, log_prob, val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(raw_action)  # 存储 Raw Action
        self.buffer.logprobs.append(log_prob)
        self.buffer.state_values.append(val)

        return proc_action.cpu().numpy().flatten()

    def update(self):
        # Update 逻辑不需要修改，因为我们是对 Logits (Raw Action) 进行优化
        # GAE 计算基于 Reward (由 Processed Action 产生) 和 Value
        # 梯度传播会自动穿过 evaluate 中的 Gaussian 分布

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals
        returns = []
        gae = 0

        # GAE Calculation
        for i in range(len(rewards) - 1, -1, -1):
            if is_terminals[i]:
                next_value = 0
                mask = 0
            else:
                if i == len(rewards) - 1:
                    next_value = old_state_values[i].item()
                else:
                    next_value = old_state_values[i + 1].item()
                mask = 1

            delta = rewards[i] + self.gamma * next_value * mask - old_state_values[i].item()
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            returns.insert(0, gae + old_state_values[i].item())

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        advantages = returns - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        avg_loss_actor = 0
        avg_loss_critic = 0

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_critic = self.MseLoss(state_values, returns)
            loss_actor = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            total_loss = loss_actor + 0.5 * loss_critic

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            avg_loss_actor += loss_actor.item()
            avg_loss_critic += loss_critic.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return {
            'loss_actor': avg_loss_actor / self.K_epochs,
            'loss_critic': avg_loss_critic / self.K_epochs
        }

    def get_lr_dict(self):
        return {'lr': self.optimizer.param_groups[0]['lr']}

    def save(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path):
        self.policy_old.load_state_dict(torch.load(path))
        self.policy.load_state_dict(torch.load(path))