import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BaselineBuffer:
    def __init__(self):
        self.actions_cont_raw = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions_cont_raw[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class BaselineActorCritic(nn.Module):
    def __init__(self, state_dim, n_dn, mode='contract', action_std_init=0.6):
        super(BaselineActorCritic, self).__init__()
        self.n_dn = n_dn
        self.mode = mode

        # --- 动作维度定义 (修改点 1) ---
        if self.mode == 'contract':
            # Contract: Dn (N+1, 差分+Scale) + W (N)
            self.dim_dn_raw = n_dn + 1  # Deltas + Scale
            self.dim_bw = n_dn
            self.act_dim = self.dim_dn_raw + self.dim_bw

        elif self.mode == 'pricing':
            # Pricing: Dn (N+1) + Rn (N+1) + W (N)
            # 假设定价模式也想用差分生成单调价格，或者保持独立
            # 这里为了简单，定价模式通常保持独立 Sigmoid，或者也用差分
            # 如果您希望 Pricing 也是单调的，可以都用 N+1
            # 这里仅演示 Contract 模式的修改，Pricing 保持原样 (N)
            self.dim_dn_raw = n_dn  # 保持独立
            self.act_dim = n_dn + n_dn + n_dn

            # --- 网络结构 ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh()
        )

        self.actor_cont = layer_init(nn.Linear(64, self.act_dim), std=0.01)
        self.action_var = nn.Parameter(torch.full((self.act_dim,), action_std_init * action_std_init))

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        features = self.actor_body(state)
        action_mean = self.actor_cont(features)

        # 采样逻辑
        action_std = torch.nn.functional.softplus(self.action_var) + 1e-5
        action_var_exp = action_std.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var_exp).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action)

        # --- 后处理 (修改点 2: 植入差分逻辑) ---
        if self.mode == 'contract':
            # 切片索引
            idx_dn = self.dim_dn_raw  # N + 1

            # --- A. 数据合同 (Dn): 差分 + 缩放 ---
            dn_raw_part = raw_action[..., :idx_dn]

            # A1. Deltas (前 N 位) -> Softplus
            deltas = F.softplus(dn_raw_part[..., :-1])

            # A2. Scale (最后 1 位) -> Sigmoid
            scale = torch.sigmoid(dn_raw_part[..., -1:])

            # A3. Cumsum (单调递增)
            shape_curve = torch.cumsum(deltas, dim=-1)

            # A4. Normalize (归一化到 0~1)
            max_val = shape_curve[..., -1:].clone()
            # 避免除以极小值
            max_val[max_val < 1e-6] = 1.0

            # 最终 Dn (0~1)
            dn_proc = (shape_curve / max_val) * scale

            # --- B. 带宽 (W): Softmax ---
            bw_proc = torch.softmax(raw_action[..., idx_dn:], dim=-1)

            processed = torch.cat([dn_proc, bw_proc], dim=-1)

        elif self.mode == 'pricing':
            # Pricing 模式保持简单的 Sigmoid 独立输出
            dn = torch.sigmoid(raw_action[..., :self.n_dn])
            rn = torch.sigmoid(raw_action[..., self.n_dn: 2 * self.n_dn])
            bw = torch.softmax(raw_action[..., 2 * self.n_dn:], dim=-1)
            processed = torch.cat([dn, rn, bw], dim=-1)

        return processed.detach(), raw_action.detach(), log_prob.detach(), self.critic(state).detach()

    def evaluate(self, state, raw_action):
        features = self.actor_body(state)
        action_mean = self.actor_cont(features)

        action_std = torch.nn.functional.softplus(self.action_var) + 1e-5
        action_var_exp = action_std.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var_exp).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        log_prob = dist.log_prob(raw_action)
        dist_entropy = dist.entropy()

        return log_prob, self.critic(state), dist_entropy


class PPO_FL:
    def __init__(self, state_dim, n_dn, mode, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.mode = mode

        self.buffer = BaselineBuffer()

        self.policy = BaselineActorCritic(state_dim, n_dn, mode).to(device)
        self.MseLoss = nn.MSELoss()

        self.optimizer = optim.Adam([
            {'params': self.policy.actor_body.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_cont.parameters(), 'lr': lr_actor},
            {'params': self.policy.action_var, 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = BaselineActorCritic(state_dim, n_dn, mode).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if state.dim() == 1: state = state.unsqueeze(0)

            proc, raw, log_prob, val = self.policy_old.act(state)

        self.buffer.states.append(state)
        # 注意: 存入的是 raw_action (包含 delta 和 scale 的原始值)
        self.buffer.actions_cont_raw.append(raw)
        self.buffer.logprobs.append(log_prob)
        self.buffer.state_values.append(val)

        return proc.cpu().numpy().flatten()

    def update(self):
        # 1. Prepare Data
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions_cont_raw, dim=0)).detach().to(device)
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
                    next_value = old_state_values[i].item()  # 边界保护
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

        # 3. Update Loop
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Loss: Critic + Actor
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

    def save(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path):
        self.policy_old.load_state_dict(torch.load(path))