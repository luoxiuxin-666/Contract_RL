import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RolloutBuffer:
    def __init__(self):
        self.actions_cont_raw = []
        self.actions_disc = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions_cont_raw[:]
        del self.actions_disc[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class HybridActorCritic(nn.Module):
    def __init__(self, state_dim, n_dn, n_cn, action_std_init=0.6):
        super(HybridActorCritic, self).__init__()
        self.n_dn = n_dn
        self.n_cn = n_cn

        # =====================================================
        # 1. 动作维度定义 (差分输出版)
        # =====================================================

        # A. 数据合同 (DN): 混合输出 (Shape + Scale)
        # 输出: N 个 Deltas (形状) + 1 个 Scale (大小)
        self.dim_dn_raw = n_dn + 1
        self.dim_dn_phys = n_dn  # 物理输出还是 N 个

        # B. 算力合同 (CN): 频率 fm_ratio
        self.dim_cn = n_cn

        # C. 带宽分配 (BW)
        self.dim_bw = n_dn + n_cn

        # 总连续动作维度 (Raw Logits)
        self.cont_action_dim = self.dim_dn_raw + self.dim_cn + self.dim_bw

        # =====================================================
        # 2. 网络结构
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

        # Actor Head (Continuous)
        # 输出 raw logits, 包含 dn_deltas, dn_scale, cn_fm, bw
        self.actor_cont = layer_init(nn.Linear(64, self.cont_action_dim), std=0.01)
        self.action_var = nn.Parameter(torch.full((self.cont_action_dim,), action_std_init * action_std_init))

        # Actor Head (Discrete)
        self.actor_disc = layer_init(nn.Linear(64, n_dn * n_cn), std=0.01)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        features = self.actor_body(state)

        # --- 1. 连续动作采样 ---
        action_mean = self.actor_cont(features)

        # 1. 使用 softplus 或 exp 确保为正
        # 2. 加上一个微小的 epsilon 防止为 0
        action_std = torch.nn.functional.softplus(self.action_var) + 1e-5
        # 扩展维度以匹配 batch
        action_var = action_std.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist_cont = MultivariateNormal(action_mean, cov_mat)

        raw_action_cont = dist_cont.sample()
        log_prob_cont = dist_cont.log_prob(raw_action_cont)

        # --- 2. 后处理: 差分生成逻辑 (Differential Generation) ---

        # 切片索引
        idx_dn_end = self.dim_dn_raw
        idx_cn_end = idx_dn_end + self.dim_cn

        # A. 数据合同 (DN): 差分 + 缩放 -> 单调递增序列
        dn_raw_part = raw_action_cont[..., :idx_dn_end]

        # A1. 提取 Deltas (前 N 位) -> 必须 > 0 -> Softplus
        deltas = F.softplus(dn_raw_part[..., :-1])

        # A2. 提取 Scale (最后 1 位) -> (0, 1) -> Sigmoid
        scale = torch.sigmoid(dn_raw_part[..., -1:])

        # A3. 生成形状 (Cumsum)
        # [d0, d0+d1, d0+d1+d2...] -> 单调递增
        shape_curve = torch.cumsum(deltas, dim=-1)

        # A4. 归一化形状 (让最大值为 1.0)
        # 加上 1e-6 防止除零
        max_val = shape_curve[..., -1:] + 1e-6
        normalized_shape = shape_curve / max_val

        # A5. 最终物理值 (0 ~ 1.0)
        # 这里的 dn_proc 已经是单调递增的了，代表 [0, MAX_DATA] 的比例
        dn_proc = normalized_shape * scale

        # B. 算力合同 (CN): fm ratio -> Sigmoid
        cn_raw = raw_action_cont[..., idx_dn_end: idx_cn_end]
        cn_proc = torch.sigmoid(cn_raw)

        # C. 带宽 (BW): Softmax
        bw_raw = raw_action_cont[..., idx_cn_end:]
        bw_proc = torch.softmax(bw_raw, dim=-1)

        # 拼接物理动作
        processed_action_cont = torch.cat([dn_proc, cn_proc, bw_proc], dim=-1)

        # --- 3. 离散动作采样 ---
        logits = self.actor_disc(features).view(-1, self.n_dn, self.n_cn)
        dist_disc = Categorical(logits=logits)
        action_disc = dist_disc.sample()
        log_prob_disc = dist_disc.log_prob(action_disc).sum(dim=-1)

        return processed_action_cont.detach(), raw_action_cont.detach(), \
            action_disc.detach(), (log_prob_cont + log_prob_disc).detach(), \
            self.critic(state).detach()

    def evaluate(self, state, raw_action_cont, action_disc):
        """
        Update 阶段: 计算 Log Prob
        注意: 这里只计算概率密度，不需要重新生成物理动作
        """
        features = self.actor_body(state)

        action_mean = self.actor_cont(features)

        action_std = torch.nn.functional.softplus(self.action_var) + 1e-5
        action_var = action_std.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist_cont = MultivariateNormal(action_mean, cov_mat)

        log_prob_cont = dist_cont.log_prob(raw_action_cont)
        dist_entropy_cont = dist_cont.entropy()

        logits = self.actor_disc(features).view(-1, self.n_dn, self.n_cn)
        dist_disc = Categorical(logits=logits)

        log_prob_disc = dist_disc.log_prob(action_disc).sum(dim=1)
        dist_entropy_disc = dist_disc.entropy().sum(dim=1)

        return log_prob_cont + log_prob_disc, self.critic(state), dist_entropy_cont + dist_entropy_disc


class PPO:
    def __init__(self, state_dim, n_dn, n_cn, lr_actor_cont, lr_actor_disc, lr_critic, gamma, K_epochs, eps_clip,
                 gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.buffer = RolloutBuffer()
        self.policy = HybridActorCritic(state_dim, n_dn, n_cn).to(device)
        self.MseLoss = nn.MSELoss()

        # =====================================================
        # 作用: 方便 LR Scheduler 统一管理 Actor 和 Critic
        # =====================================================
        self.optimizer = optim.Adam([
            # --- Actor 参数组 ---
            # Body: 基础 LR
            {'params': self.policy.actor_body.parameters(), 'lr': min(lr_actor_cont, lr_actor_disc)},
            # Cont Head: 连续动作 LR
            {'params': self.policy.actor_cont.parameters(), 'lr': lr_actor_cont},
            {'params': self.policy.action_var, 'lr': lr_actor_cont},
            # Disc Head: 离散动作 LR
            {'params': self.policy.actor_disc.parameters(), 'lr': lr_actor_disc},

            # --- Critic 参数组 ---
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = HybridActorCritic(state_dim, n_dn, n_cn).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if state.dim() == 1: state = state.unsqueeze(0)

            proc_cont, raw_cont, disc, log_prob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions_cont_raw.append(raw_cont)
        self.buffer.actions_disc.append(disc)
        self.buffer.logprobs.append(log_prob)
        self.buffer.state_values.append(state_val)

        return proc_cont.cpu().numpy().flatten(), disc.cpu().numpy().flatten()

    def update(self):
        # 1. 转换数据
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions_cont = torch.squeeze(torch.stack(self.buffer.actions_cont_raw, dim=0)).detach().to(device)
        old_actions_disc = torch.squeeze(torch.stack(self.buffer.actions_disc, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals

        # 2. 计算 GAE (Generalized Advantage Estimation)
        returns = []
        gae = 0
        for i in range(len(rewards) - 1, -1, -1):
            if is_terminals[i]:
                next_value = 0
                mask = 0
            else:
                # 边界保护
                if i == len(rewards) - 1:
                    next_value = old_state_values[i].item()
                else:
                    next_value = old_state_values[i + 1].item()
                mask = 1

            delta = rewards[i] + self.gamma * next_value * mask - old_state_values[i].item()
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            returns.insert(0, gae + old_state_values[i].item())

        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # =====================================================
        # 核心修改 B: Reward Normalization
        # 作用: 即使环境 Reward 是 1500，这里也会变成 ~0.5，防止 Loss 爆炸
        # =====================================================
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # 重新计算优势 (基于归一化后的 Returns)
        # 注意: 这里 old_state_values 虽然是基于旧 Reward 估算的，
        # 但我们主要关心的是 Advantage 的相对大小。
        # 更严谨的做法是：Critic 也应该学会输出归一化后的 Value。
        # 但在 PPO 实践中，直接对 Returns 归一化然后算 Adv 是最常用的 Trick。
        advantages = returns - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 记录 Loss
        avg_loss_actor = 0
        avg_loss_critic = 0

        # 3. Update Loop
        for _ in range(self.K_epochs):
            # 评估新策略
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions_cont, old_actions_disc)
            state_values = torch.squeeze(state_values)

            # Ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # --- Loss Calculation ---
            # Actor Loss
            loss_actor = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()

            # Critic Loss (MSE)
            loss_critic = self.MseLoss(state_values, returns)

            # =====================================================
            # 核心修改 C: 联合反向传播 (Joint Backward)
            # =====================================================
            # 权重系数: 0.5 是 Critic 的标准权重
            total_loss = loss_actor + 0.5 * loss_critic

            self.optimizer.zero_grad()
            total_loss.backward()

            # 全局梯度裁剪
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

            self.optimizer.step()

            # 记录 (用于绘图)
            avg_loss_actor += loss_actor.item()
            avg_loss_critic += loss_critic.item()

        # 4. 同步 & 清空
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        # 返回平均 Loss
        return {
            'loss_actor': avg_loss_actor / self.K_epochs,
            'loss_critic': avg_loss_critic / self.K_epochs
        }

    def get_lr_dict(self):
        """
        返回不同组件的学习率 (由于现在是同一个 optimizer，从 param_groups 取)
        """
        return {
            'cont': self.optimizer.param_groups[1]['lr'],  # Index 1 是 Actor Cont Head
            'disc': self.optimizer.param_groups[3]['lr'],  # Index 3 是 Actor Disc Head
            'critic': self.optimizer.param_groups[4]['lr']  # Index 4 是 Critic
        }

    def save(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path):
        self.policy_old.load_state_dict(torch.load(path))
        self.policy.load_state_dict(torch.load(path))