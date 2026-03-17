import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# 自动选择计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ==============================================================================
# 1. 经验池 (Rollout Buffer)
# ==============================================================================
class PricingBuffer:
    def __init__(self):
        self.actions = []  # 存储原始的单价输出
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


# ==============================================================================
# 2. PPO 神经网络架构 (仅输出 1 个统一定价参数)
# ==============================================================================
class UniformPricingActorCritic(nn.Module):
    def __init__(self, state_dim, n_dn, action_std_init=0.6):
        super(UniformPricingActorCritic, self).__init__()
        self.n_dn = n_dn

        # --- 动作维度 ---
        # 1. p_dn_ratio: 给所有数据节点的统一基准单价比例 [0, 1]
        # (因为是 FL 架构，不需要算力节点，所以只有 1 个输出)
        self.action_dim = 1

        # 共享特征提取网络
        self.shared_net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh()
        )

        # Critic 网络
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # Actor 网络
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
            nn.Sigmoid()  # 将价格比例限制在 [0, 1] 之间
        )

        # 动作标准差参数 (用于探索)
        self.actor_log_std = nn.Parameter(torch.zeros(self.action_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        features = self.shared_net(state)
        action_mean = self.actor_mean(features)

        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)

        raw_action = dist.sample()

        # 防止加了噪声之后数值超出 [0, 1]
        action_clipped = torch.clamp(raw_action, 0.001, 0.999)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        state_val = self.critic(features)

        return action_clipped.detach(), raw_action.detach(), log_prob.detach(), state_val.detach()

    def evaluate(self, state, raw_action):
        features = self.shared_net(state)
        action_mean = self.actor_mean(features)

        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(features)

        return log_prob, state_values, dist_entropy


# ==============================================================================
# 3. PPO 算法核心类
# ==============================================================================
class PPO_FL_UniformPricing:
    def __init__(self, state_dim, n_dn, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.buffer = PricingBuffer()

        self.policy = UniformPricingActorCritic(state_dim, n_dn).to(device)
        self.MseLoss = nn.MSELoss()

        self.optimizer = optim.Adam([
            {'params': self.policy.shared_net.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_log_std, 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = UniformPricingActorCritic(state_dim, n_dn).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if state.dim() == 1: state = state.unsqueeze(0)

            proc_action, raw_action, log_prob, val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(raw_action)
        self.buffer.logprobs.append(log_prob)
        self.buffer.state_values.append(val)

        return proc_action.cpu().numpy().flatten()

    def update(self):
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals

        # --- GAE 计算 ---
        returns = []
        gae = 0
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
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        advantages = returns - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        avg_loss_a = 0
        avg_loss_c = 0

        # --- PPO 更新循环 ---
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

            avg_loss_a += loss_actor.item()
            avg_loss_c += loss_critic.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return {
            'loss_actor': avg_loss_a / self.K_epochs,
            'loss_critic': avg_loss_c / self.K_epochs
        }

    def get_lr_dict(self):
        return {
            'actor': self.optimizer.param_groups[1]['lr'],
            'critic': self.optimizer.param_groups[3]['lr']
        }

    def save(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path):
        self.policy_old.load_state_dict(torch.load(path))

def decode_action_fl_pricing(ppo_action, env):
    """
    PPO 仅输出 1 个标量：[p_dn_ratio] (统一单价比例)
    """
    cfg = env.cfg
    N = cfg.N_DN

    # 1. 映射为真实的全局统一单价 (元/单位数据)
    # 假设 FL 模式下，单价的上限要设得比 SFL 高得多，否则没人干活
    Diff = env.DN_list[-1].unit_cost - env.DN_list[0].unit_cost
    Base = env.DN_list[0].unit_cost  # 例如 100.0
    # P_DN_MAX = 2*env.DN_list[0].unit_cost  # 例如 100.0

    # p_dn_global = ppo_action[0] * P_DN_MAX
    p_dn_global =  Base + ppo_action[0] * 2 * Diff

    Dn_phys = np.zeros(N)
    Rn_phys = np.zeros(N)
    W_alloc = np.ones(N) * (cfg.TOTAL_BW / N)  # FL 中默认带宽平分，不优化

    # =========================================================
    # 2. 数据节点理性响应 (基于统一定价)
    # =========================================================
    C_FATIGUE = 1e-5

    for i, dn in enumerate(env.DN_list):
        c_n_base = dn.unit_cost  # 0.114

        # 只有出价大于保本价，才有动力接单
        if p_dn_global > c_n_base:
            # 每超出保本价一点点，数据量就会大幅增加
            d_opt = (p_dn_global - c_n_base) / (2.0 * C_FATIGUE)
        else:
            d_opt = 0.0

        d_opt = np.clip(d_opt, 0, env.cfg.max_data_count)

        Dn_phys[i] = d_opt
        Rn_phys[i] = p_dn_global * d_opt

    return {
        'pricing': p_dn_global,
        'Dn': Dn_phys,
        'Rn': Rn_phys,
        'bandwidth': W_alloc,
        'mode': 'FL_PRICING_BASELINE'  # 让环境 step 知道此时是 FL 模式，计算效用不要带 CN
    }

def pricing_run_training(env,ppo,ppo_state):
    current_steps = 0
    state = ppo_state
    episode_reward = []
    episode_time = []
    episode_total_data = []
    episode_pricing = []
    while True:
        current_steps += 1
        # 2.1 Agent 选择动作 (归一化值 + 离散索引)
        proc_cont = ppo.select_action(state)
        action = decode_action_fl_pricing(proc_cont, env)
        episode_pricing.append(action['pricing'])
        # 2.2 环境交互
        # 环境内部会进行 IC/IR 检查、能耗计算、奖励计算
        # return next_state, reward, False, contract, total_data
        next_state, reward, done, contract, total_data = env.step3(action)

        # 2.3 判断终止条件 (Termination vs Truncation)
        # A. 任务失败/完成 (Done)
        # B. 达到最大步数 (Time Limit)
        time_limit_reached = (current_steps >= env.cfg.MAX_STEPS_PER_EP)
        is_terminal = done or time_limit_reached

        # 2.5 存储经验
        # 存入的是 is_terminal，PPO update 时会据此截断 GAE
        ppo.buffer.rewards.append(reward)
        ppo.buffer.is_terminals.append(is_terminal)

        state = next_state
        episode_reward.append(reward)
        episode_time.append(env.uav.max_time)
        episode_total_data.append(total_data)

        # 2.6 终止判断
        if is_terminal:
            break

    return episode_reward, episode_time ,episode_total_data, episode_pricing