from copy import deepcopy

import numpy as np
import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
from plot_picture import plot_learning_curves
from tradition_contract.sfl_contract2 import TraditionalContractBaseline
from uniform_pricing.sfl_ppo_pricing import pricing_run_training, UniformPricingPPO
# 导入自定义模块
from UsualFunctions import LOG  # 假设这是您的日志工具
from Contract_Env_2 import Contract_Environment
from Cont_PPO import PPO
from Contract_Config import Config

# ==========================================
# 0. 日志与设备设置
# ==========================================
log = LOG()
log.LogInitialize()


def Log(message, Flag=True):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.LogRecord(f"{message} - 时间：{current_time}", Flag)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")


# ==========================================
# 1. 主执行器类
# ==========================================
class SFLExecutionRunner:
    def __init__(self, config):
        self.cfg = config
        self.SFL_CONTRACT = self.cfg.SFL_CONTRACT
        self.PPO_PRICING = self.cfg.PPO_PRICING
        # 1.1 初始化环境
        # 环境会读取 config 中的 N_DN, M_CN 等参数
        self.env = Contract_Environment(self.cfg)

        # 是否开启其他对照实验
        if self.cfg.SFL_CONTRACT:
            self.env_sfl_contract = deepcopy(self.env)
        if self.cfg.PPO_PRICING:
            # state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda=0.95
            self.env_ppo_pricing = deepcopy(self.env)
            pricing_state_dim = self.env_ppo_pricing.state_dim
            self.uniform_ppo = UniformPricingPPO(
                state_dim=pricing_state_dim,
                lr_actor=config.LR_ACTOR,  # 连续动作学习率 (大)
                lr_critic=config.LR_CRITIC,
                gamma=config.GAMMA,
                K_epochs=config.K_EPOCHS,
                eps_clip=config.EPS_CLIP)

            if self.cfg.USE_LR_SCHEDULER:
                self.scheduler_pricing = ReduceLROnPlateau(
                    self.uniform_ppo.optimizer,  # 监控它的优化器
                    mode='max',  # 目标是 Reward 最大化
                    factor=self.cfg.LR_FACTOR,
                    patience=self.cfg.LR_PATIENCE,
                    threshold=self.cfg.LR_THRESHOLD,
                    min_lr=self.cfg.LR_MIN,
                    verbose=True
                )

        self.contract = TraditionalContractBaseline(self.cfg)
        # 1.2 初始化 PPO Agent
        # 注意: 传入分层学习率参数 (lr_actor_cont, lr_actor_disc)
        # 假设 Config 类中已经定义了这些参数，如果没有，请在 Config 中添加
        # state_dim, n_dn, n_cn, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda=0.95

        self.agent = PPO(
            state_dim=self.env.state_dim,
            n_dn=config.N_DN,
            n_cn=config.M_CN,
            lr_actor=config.LR_ACTOR,  # 连续动作学习率 (大)
            lr_critic=config.LR_CRITIC,
            gamma=config.GAMMA,
            K_epochs=config.K_EPOCHS,
            eps_clip=config.EPS_CLIP
        )
        # 1.3 初始化动态学习率调度器
        if self.cfg.USE_LR_SCHEDULER:
            # 修改点：监控统一的 self.agent.optimizer
            self.scheduler = ReduceLROnPlateau(
                self.agent.optimizer,
                mode='max',
                factor=self.cfg.LR_FACTOR,
                patience=self.cfg.LR_PATIENCE,
                threshold=self.cfg.LR_THRESHOLD,
                min_lr=self.cfg.LR_MIN
            )

        self.metrics = {
            'Total_Data': [],  # 这里的 Reward 已经是平滑过的了，但为了画图再存一份
            'Avg_Reward': [],  # 平均reward
            'Avg_Latency': [],  # 系统时延
            'SFL_Contract_Total_Data': [],  # 系统时延
            'SFL_Contract_Uti': [],  # 普通合同unit
            'SFL_Contract_Time': [],  # 监控 LR 变化
            'Pricing_Total_Data': [],  # 如果 PPO 返回 Loss
            'Pricing_Uti': [],
            'Pricing_Latency': []
        }

    def run_training(self):
        print(f"--- Starting Training (Max Episodes: {self.cfg.TOTAL_EPISODES}) ---")
        Log("训练开始")
        log_max_reward = {
            'max_reward': 0.0,
            'i_episode': 0,
            'dn_contract': [],
            'cn_contract': [],
            'total_time': 0.0,
            'cn_uti': 0.0,
            'dn_uti': 0.0,
            'uav_cost': 0.0,
        }
        all_log_max_reward = []
        for i_episode in range(1, self.cfg.TOTAL_EPISODES + 1):
            # 1. 重置环境，获取初始状态
            state = self.env.reset()
            episode_reward = []
            episode_time = []
            episode_uav_cost = []
            episode_total_data = []
            ep_reward = 0
            # 2. Episode 步进循环
            # SFL 通常是一轮决策，所以这里的 steps 可能就是 1，
            # 或者如果是多轮连续训练，steps 就是 global rounds
            current_steps = 0
            while True:
                current_steps += 1
                # 2.1 Agent 选择动作 (归一化值 + 离散索引)
                proc_cont = self.agent.select_action(state)

                # 2.2 环境交互
                # 环境内部会进行 IC/IR 检查、能耗计算、奖励计算
                next_state, reward, done, uav_info, dn_contract, cn_contract, uti_, total_data = self.env.step(
                    proc_cont)

                W_all = np.round(proc_cont[self.env.N_DN + self.env.M_CN:-self.env.M_CN], 3) * (self.env.TOTAL_BW / 1e6)

                # 2.3 判断终止条件 (Termination vs Truncation)
                # A. 任务失败/完成 (Done)
                # B. 达到最大步数 (Time Limit)
                time_limit_reached = (current_steps >= self.cfg.MAX_STEPS_PER_EP)
                is_terminal = done or time_limit_reached

                # 2.5 存储经验
                # 存入的是 is_terminal，PPO update 时会据此截断 GAE
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(is_terminal)

                state = next_state
                ep_reward += reward
                episode_reward.append(reward)
                episode_time.append(uav_info['total_time'])  # 假设 ep_time 是本轮总耗时
                episode_uav_cost.append(uav_info['total_cost'])
                episode_total_data.append(total_data)

                if reward > log_max_reward['max_reward']:
                    log_max_reward = {
                        'max_reward': reward,
                        'i_episode': i_episode,
                        'dn_contract': dn_contract,
                        'cn_contract': cn_contract,
                        'total_time': episode_time[-1],
                        'cn_uti': uti_['cn_uti'],
                        'dn_uti': uti_['dn_uti'],
                        'uav_cost': uti_['uav_uti'],
                    }
                # 2.6 终止判断
                if is_terminal:
                    break

            # 3. Episode 结束处理
            if i_episode >= 500 and i_episode % 500 == 0:
                all_log_max_reward.append(log_max_reward)
                log_max_reward = {
                    'max_reward': 0.0,
                    'i_episode': 0,
                    'dn_contract': [],
                    'cn_contract': [],
                    'total_time': 0.0,
                    'cn_uti': 0.0,
                    'dn_uti': 0.0,
                    'uav_cost': 0.0,
                }
                for log_reward in all_log_max_reward:
                    Log(log_reward, False)
            # 3.1 更新 PPO 策略
            # PPO 是 On-policy 算法，通常在收集完一个完整的 Episode (或一定数量的 Steps) 后更新
            all_loss = self.agent.update()

            # 3.2 记录与调度
            avg_reward = np.mean(episode_reward)
            avg_time = np.mean(episode_time)
            true_time = float(episode_time[-1])
            avg_uav_cost = np.mean(episode_uav_cost)
            avg_total_data = np.mean(episode_total_data)
            avg_loss_actor = np.array(all_loss['loss_actor'])
            avg_loss_critic = np.array(all_loss['loss_critic'])
            # --- 2. 收集数据 ---
            # 这些数据通常是本轮 Episode 的统计值
            self.metrics['Total_Data'].append(avg_total_data)
            self.metrics['Avg_Reward'].append(avg_reward)
            self.metrics['Avg_Latency'].append(avg_time)
            # self.metrics['Actor_Loss'].append(avg_loss_actor)
            # self.metrics['Critic_Loss'].append(avg_loss_critic)
            # self.metrics['UAV_Cost'].append(avg_uav_cost)

            if self.SFL_CONTRACT:
                self.env_sfl_contract.reset()
                contract_action = self.contract.get_action(self.env_sfl_contract)
                state_contract, contract_reward, done_contract, uav_info_contract, dn_cont, cn_cont, uti_contract, all_data = self.env_sfl_contract.step2(
                    contract_action)
                self.metrics['SFL_Contract_Uti'].append(contract_reward)
                self.metrics['SFL_Contract_Time'].append(uav_info['total_time'])
                self.metrics['SFL_Contract_Total_Data'].append(all_data)

            if self.PPO_PRICING:
                self.env_ppo_pricing.reset()
                # env,ppo,ppo_state
                pricing_state = self.env_ppo_pricing.state
                episode_reward, episode_time, episode_total_data = pricing_run_training(self.env_ppo_pricing,self.uniform_ppo,pricing_state)
                all_loss = self.uniform_ppo.update()
                self.metrics['Pricing_Total_Data'].append(np.mean(episode_total_data))
                self.metrics['Pricing_Uti'].append(np.mean(episode_reward))
                self.metrics['Pricing_Latency'].append(np.mean(episode_time))

            # 3.3 动态调整学习率 (基于滑动平均奖励)
            if i_episode >= 500:
                if self.cfg.USE_LR_SCHEDULER:
                    self.scheduler.step(avg_reward)

                    if self.PPO_PRICING:
                        self.scheduler_pricing.step(np.mean(episode_reward))


            # 3.4 日志打印
            if i_episode % self.cfg.LOG_INTERVAL == 0:
                # 获取不同部分的 LR 用于监控
                lrs = self.agent.get_lr_dict() if hasattr(self.agent, 'get_lr_dict') else {'actor': 0}
                log_msg = f"Ep {i_episode} | Reward: {avg_reward:.2f} |  LR_Cont: {lrs.get('lr', 0):.2e}|true_time: {true_time:.2f}|avg_time: {avg_time:.2f}|avg_cost: {avg_uav_cost:.2f}"
                Log(log_msg)
                log_msg = f"uav_uti: {uti_['uav_uti']:.2f}|dn_uti: {uti_['dn_uti']:.2f}|cn_uti: {uti_['cn_uti']:.2f}"
                Log(log_msg, False)
                msg = f"dn_contract {dn_contract} | cn_contract {cn_contract}"
                Log(msg, False)
                msg = f" W_all is {W_all}"
                Log(msg, False)
                # if compliance_rate!=1:
                #     Log(f"============== compliance_rate is 1 =================",False)

            # --- 3. 调用绘图 ---
            # 不需要每轮都画，太慢了。每 LOG_INTERVAL 画一次即可。
            if i_episode % self.cfg.LOG_INTERVAL == 0:
                # 调用我们写好的改进版绘图函数
                # 注意：传入当前的 i_episode，函数会自动计算横坐标
                plot_learning_curves(self.metrics, i_episode, window_size=20)

            # 3.5 保存模型
            if i_episode % self.cfg.SAVE_INTERVAL == 0:
                save_path = f"./checkpoints/ep_{i_episode}.pth"
                if not os.path.exists("./checkpoints"): os.makedirs("./checkpoints")
                self.agent.save(save_path)
                Log(f"模型保存: {save_path}")

            # if i_episode % 1000 == 0:
            #     print(f"---the cn is {self.env.uav.data}")


# ==========================================
# 2. 程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 加载配置
    # 这里直接实例化 Config 类，确保包含所有必要参数
    cfg = Config()

    # 2. 实例化运行器
    runner = SFLExecutionRunner(cfg)

    # 3. 开始训练
    runner.run_training()

    Log("训练结束")
