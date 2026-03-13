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
# 1. 简化的经验池 Buffer
# ==============================================================================
class PricingBuffer:
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


# ==============================================================================
# 2. PPO 神经网络架构 (仅输出 2 个定价参数)
# ==============================================================================
class UniformPricingActorCritic(nn.Module):
    def __init__(self, state_dim, action_std_init=0.6):
        super(UniformPricingActorCritic, self).__init__()

        # 动作维度 = 2
        # Action[0] = p_dn_ratio (数据节点基准单价比例)
        # Action[1] = p_cn_ratio (算力节点基准单价比例)
        self.action_dim = 2

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
        self.actor_log_std = nn.Parameter(torch.full((self.action_dim,), np.log(action_std_init)))

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        features = self.shared_net(state)
        action_mean = self.actor_mean(features)

        action_std = torch.exp(self.actor_log_std)
        # 因为各维度独立，使用标准的 Normal 即可，无需 MultivariateNormal
        dist = Normal(action_mean, action_std)

        action = dist.sample()

        # 防止加了噪声之后数值超出 [0, 1] 导致价格为负或暴涨
        action_clipped = torch.clamp(action, 0.001, 0.999)

        # 计算 log_prob (各维度求和)
        log_prob = dist.log_prob(action).sum(dim=-1)

        state_val = self.critic(features)

        return action_clipped.detach(), action.detach(), log_prob.detach(), state_val.detach()

    def evaluate(self, state, action):
        features = self.shared_net(state)
        action_mean = self.actor_mean(features)

        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)

        state_values = self.critic(features)

        return log_prob, state_values, dist_entropy


# ==============================================================================
# 3. PPO 算法核心类
# ==============================================================================
class UniformPricingPPO:
    def __init__(self, state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.buffer = PricingBuffer()

        self.policy = UniformPricingActorCritic(state_dim).to(device)
        self.MseLoss = nn.MSELoss()

        self.optimizer = optim.Adam([
            {'params': self.policy.shared_net.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_log_std, 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = UniformPricingActorCritic(state_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if state.dim() == 1: state = state.unsqueeze(0)

            proc_action, raw_action, log_prob, val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(raw_action)  # 存原始值用于求梯度
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
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)  # 归一化

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

        return avg_loss_a / self.K_epochs, avg_loss_c / self.K_epochs


# ==============================================================================
# 4. 动作解析逻辑 (放在 Runner 里或者写成独立函数)
# ==============================================================================
def decode_action_uniform_pricing(ppo_action, env):
    """
    将 PPO 输出的 2 个价格比例，转化为完整的物理动作字典。
    """
    N_DN = env.cfg.N_DN
    M_CN = env.cfg.M_CN
    
    # 1. 提取 PPO 决定的全局统一单价
    # 假设给 DN 的最大可能单价是 100.0, 给 CN 的最大可能单价是 50.0
    P_DN_MAX = 5*env.DN_list[0].unit_cost
    P_CN_MAX = 5*(1/env.CN_list[0].type)

    p_dn_global = ppo_action[0] * P_DN_MAX
    p_cn_global = ppo_action[1] * P_CN_MAX

    # 2. 数据节点理性响应 (自组织计算 Dn)
    Dn_phys = np.zeros(N_DN)
    Rn_phys = np.zeros(N_DN)

    C_QUAD = 1e-5

    for i, dn in enumerate(env.DN_list):
        # 假设节点在最优化自己的效用: U = P_dn * D - c_n * D^2
        # 最优数据量 D* = P_dn / (2 * c_n)
        # 注意: 您的真实成本函数可能不同，请替换为实际的偏导公式
        c_n = dn.unit_cost
        # 只有单价大于基础成本，才有干活的动力
        if p_dn_global > c_n:
            # 利用二次成本算出最优数量
            optimal_d = (p_dn_global - c_n) / (2.0 * C_QUAD)
        else:
            optimal_d = 0.0

        # 物理限制
        optimal_d = np.clip(optimal_d, 0, env.MAX_DATA_COUNT)

        Dn_phys[i] = optimal_d
        Rn_phys[i] = p_dn_global * optimal_d

    # 3. 启发式路由 (贪婪负载均衡)
    beta_m = np.zeros(M_CN)
    routing = np.zeros(N_DN, dtype=int)

    # 这里计算的是相对大小不是精准的能耗
    H_list = []
    for i, dn in enumerate(env.DN_list):
        H_alpha = dn.body_flops * Dn_phys[i]
        H_list.append(np.round(H_alpha/1e3, 3))# 这里使用GFlops

    sorted_dn_desc = np.argsort(H_list)[::-1]
    for n in sorted_dn_desc:
        if H_list[n] > 0:
            lightest_cn = np.argmin(beta_m)
            routing[n] = lightest_cn
            beta_m[lightest_cn] += H_list[n]
        else:
            routing[n] = 0

    # 4. 算力节点被动响应 (计算频率和收益)
    fm_phys = np.zeros(M_CN)
    Rm_total = np.zeros(M_CN)

    TAU_REQ = 5e3  # 强制时间要求

    for m in range(M_CN):
        if beta_m[m] > 1e-5:
            # 被迫以刚好及格的速度运行以省电
            req_freq = (beta_m[m] * 1e9) / TAU_REQ
            fm_phys[m] = np.clip(req_freq, 1e9, 10e9)

            # 收到的钱 = 全局算力单价 * 任务量
            Rm_total[m] = p_cn_global * beta_m[m]
        else:
            fm_phys[m] = 1e9
            Rm_total[m] = 0.0



    # 5. 简单的带宽分配
    TOTAL_BW = env.cfg.TOTAL_BW
    W_alloc = np.zeros(N_DN + M_CN)
    total_D = np.sum(Dn_phys) + 1e-9
    for n in range(N_DN):
        W_alloc[n] = (TOTAL_BW / 2) * (Dn_phys[n] / total_D)
    for m in range(M_CN):
        W_alloc[N_DN + m] = (TOTAL_BW / 2) / M_CN

    return {
        'Dn': Dn_phys, 'Rn': Rn_phys,
        'Rm': Rm_total, 'fm': fm_phys,
        'bandwidth': W_alloc, 'routing': routing,
        'beta_m': beta_m, 'mode': 'UNIFORM_PRICING'
    }


def pricing_run_training(env,ppo,ppo_state):
    current_steps = 0
    state = ppo_state
    episode_reward = []
    episode_time = []
    episode_total_data = []
    while True:
        current_steps += 1
        # 2.1 Agent 选择动作 (归一化值 + 离散索引)
        proc_cont = ppo.select_action(state)

        action = decode_action_uniform_pricing(proc_cont, env)
        # 2.2 环境交互
        # 环境内部会进行 IC/IR 检查、能耗计算、奖励计算
        next_state, reward, done, uav_info, dn_contract, cn_contract, uti_, total_data = env.step3(action)

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
        episode_time.append(uav_info['total_time'])
        episode_total_data.append(total_data)

        # 2.6 终止判断
        if is_terminal:
            break

    return episode_reward, episode_time ,episode_total_data