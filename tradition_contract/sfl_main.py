from SFL_Contract import TraditionalContractBaseline
from Contract_Config import Config
import numpy as np
from Contract_Env_2 import Contract_Environment

# 传统合同的计算过程
def run_tct(config):
    print("--- Starting TCT Baseline ---")

    # 记录数据
    tct_rewards = []
    # 加载一下环境
    env = Contract_Environment(config)

    # TCT 是静态策略，所以其实跑一轮就够了，
    # 但由于环境中信道(SINR)和电量是动态变化的，所以跑多轮看它的崩溃情况。
    for i_episode in range(1, 100):
        state = env.reset()
        dn_list = env.DN_list
        cn_list = env.CN_list
        # TCT 不需要输入 state，它闭着眼睛发设定好的合同
        contracts = tct_agent.get_action(env)

        # 放入环境执行，环境里的物理节点会用真实的信道和电量来评估
        next_state, reward, done, info = env.step_2(contracts)

        tct_rewards.append(reward)

    print(f"TCT Average Reward: {np.mean(tct_rewards)}")


# main_runner.py 示例
if __name__ == '__main__':
    config = Config()
    # 实例化 TCT Solver
    tct_agent = TraditionalContractBaseline(config)
