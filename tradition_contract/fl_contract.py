import numpy as np


class FL_TraditionalContractBaseline:
    """
    传统联邦学习 (FL) 架构下的传统合同理论 (TCT) 求解器。
    作为 PPO 的 Baseline。
    特点:
    1. 只有数据节点 (DN)，没有算力节点 (CN) 和路由。
    2. 物理成本基于 FL 架构：本地计算全量模型 + 上传模型参数。
    3. 基于数学解析解直接计算满足 IC/IR 的最优菜单 (Dn, Rn)。
    """

    def __init__(self, env):
        self.env = env
        self.N_DN = env.N_DN

        # 假设服务器认为节点类型服从均匀分布
        self.prob_dn = np.ones(self.N_DN) / self.N_DN

        # 模型收益权重 (需与环境 Reward 对齐)
        self.BETA_1 = 500.0

    def _perform_ironing(self, virtual_costs):
        """
        池化算法 (Ironing)：确保虚拟成本 J_k 严格非递减
        """
        N = len(virtual_costs)
        j_ironed = np.copy(virtual_costs)

        is_monotonic = False
        while not is_monotonic:
            is_monotonic = True
            for k in range(N - 1):
                if j_ironed[k] > j_ironed[k + 1]:
                    avg_cost = (j_ironed[k] + j_ironed[k + 1]) / 2.0
                    j_ironed[k] = avg_cost
                    j_ironed[k + 1] = avg_cost
                    is_monotonic = False
        return j_ironed

    def get_action(self):
        """
        求解最优 FL 合同并分配
        """
        # =========================================================
        # 1. 提取真实的 FL 物理成本并排序
        # =========================================================

        # A. 从环境获取真实成本并排序 (Index 0 最便宜，Index N-1 最贵)
        dn_costs_raw = np.zeros(self.N_DN)
        for i, dn in enumerate(self.env.DN_list):
            dn_costs_raw[i] = dn.unit_cost

        # 排序：Index 0 最便宜(最好)，Index N-1 最贵(最差)
        dn_sort_idx = np.argsort(dn_costs_raw)
        dn_costs_asc = dn_costs_raw[dn_sort_idx]

        # B. 计算原始虚拟成本 (Virtual Cost: J)
        # 顺着计算 (从最好到最差)，因为租金是"好人吃差人"
        J_dn_raw = np.zeros(self.N_DN)
        cum_prob_better = 0

        for k in range(self.N_DN):
            if k == 0:
                J_dn_raw[k] = dn_costs_asc[k]
                cum_prob_better += self.prob_dn[k]
            else:
                cost_diff = dn_costs_asc[k] - dn_costs_asc[k - 1]
                hazard_rate = cum_prob_better / self.prob_dn[k]
                J_dn_raw[k] = dn_costs_asc[k] + hazard_rate * cost_diff
                cum_prob_better += self.prob_dn[k]

        # C. 执行池化 (Ironing)
        J_dn_ironed = self._perform_ironing(J_dn_raw)

        # D. 求解最优数据量 D* (解析解: A/J - 1)
        Dn_optimal = np.zeros(self.N_DN)
        for k in range(self.N_DN):
            d_star = (self.BETA_1 / (J_dn_ironed[k] + 1e-9)) - 1
            d_star = np.clip(d_star, 0, self.env.cfg.max_data_count)
            Dn_optimal[k] = np.round(d_star, 0)

        # 此时 Dn_optimal 天然是降序的 (大任务 -> 小任务)

        # E. IC 递推求解激励 Rn
        Rn_optimal = np.zeros(self.N_DN)
        U_acc = 0
        # 从最差节点 (Index N-1) 倒推
        for k in range(self.N_DN - 1, -1, -1):
            if k == self.N_DN - 1:
                Rn_optimal[k] = dn_costs_asc[k] * Dn_optimal[k]
            else:
                rent = (dn_costs_asc[k + 1] - dn_costs_asc[k]) * Dn_optimal[k + 1]
                U_acc += rent
                Rn_optimal[k] = (dn_costs_asc[k] * Dn_optimal[k]) + U_acc

        # F. 映射回物理节点顺序
        Dn_phys = np.zeros(self.N_DN)
        Rn_phys = np.zeros(self.N_DN)
        Dn_phys[dn_sort_idx] = Dn_optimal
        Rn_phys[dn_sort_idx] = Rn_optimal
        # =========================================================
        # 6. 带宽分配 (FL 特有)
        # =========================================================
        # FL 中所有 DN 都要上传等大的模型参数，为了最小化瓶颈时延，
        # 应该采用"信道倒数"的注水法，或者最简单的均匀分配。
        W_alloc = np.ones(self.N_DN) * (self.env.TOTAL_BW / self.N_DN)

        return {
            'Dn': Dn_phys,
            'Rn': Rn_phys,
            'bandwidth': W_alloc,
            'mode': 'FL_TCT_BASELINE'  # 标记，让环境以 FL 模式计算效用
        }
    
if __name__ == '__main__':
    from FL_RL.FL_Env import FLEnvironment
    from Contract_Config import Config
    config = Config()
    env = FLEnvironment(config)
    env.reset()
    experiment = FL_TraditionalContractBaseline(env)
    contract = experiment.get_action()
    # 环境内部会进行 IC/IR 检查、能耗计算、奖励计算
    next_state, reward, done, dn_contract, total_data = env.step2(contract)

    print(reward,dn_contract, total_data)
