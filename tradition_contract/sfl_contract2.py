import numpy as np
import copy


class TraditionalContractBaseline:
    """
    传统合同理论 (Traditional Contract Theory, TCT) 求解器
    作为 PPO 的 Baseline (基准对比算法)。
    包含完整的: 虚拟成本计算、池化(Ironing)算法、最优频率解析解、以及事后重匹配定价。
    """

    def __init__(self, config):
        self.cfg = config
        self.N_DN = config.N_DN
        self.M_CN = config.M_CN

        # 假设服务器拥有对节点类型的“先验知识”(Prior Knowledge)
        # 这里假设为均匀分布
        self.prob_dn = np.ones(self.N_DN) / self.N_DN
        self.prob_cn = np.ones(self.M_CN) / self.M_CN

        # 假设模型精度收益系数 (需与环境 Reward 计算对齐)
        # self.BETA_1 = self.cfg.alpha_dn  # 收益系数
        self.BETA_1 = 450  # 收益系数
        self.ETA = 1.0  # 时延惩罚权重
        self.MU_COEFF = 1e-28  # 能耗常数
        self.E_COMM_UNIT = 0.01  # 单位传输常数

    def _perform_ironing(self, virtual_costs):
        """
        核心池化算法 (Ironing Algorithm)
        目标: 确保虚拟成本 J_k 是严格非递减的 (即 J_0 <= J_1 <= ... <= J_{N-1})
        """
        N = len(virtual_costs)
        j_ironed = np.copy(virtual_costs)

        is_monotonic = False
        while not is_monotonic:
            is_monotonic = True
            for k in range(N - 1):
                # 如果前一个(更好的类型)的虚拟成本，竟然比后一个(更差的类型)还高
                if j_ironed[k] > j_ironed[k + 1]:
                    # 合并取平均值
                    avg_cost = (j_ironed[k] + j_ironed[k + 1]) / 2.0
                    j_ironed[k] = avg_cost
                    j_ironed[k + 1] = avg_cost
                    is_monotonic = False

        return j_ironed

    def get_action(self, env):
        """
        求解最优合同并分配，返回与 PPO 相同格式的 action_dict
        """

        # =========================================================
        # 1. 求解数据节点合同 (DN Contract)
        # =========================================================

        # A. 从环境获取真实成本并排序 (Index 0 最便宜，Index N-1 最贵)
        dn_costs_raw = np.zeros(self.N_DN)
        for i, dn in enumerate(env.DN_list):
            dn_costs_raw[i] = dn.unit_cost

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
            d_star = np.clip(d_star, 0, self.cfg.max_data_count)
            Dn_optimal[k] = np.round(d_star,0)

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
        # 2. 启发式路由 (Heuristic Routing) - 盲分
        # =========================================================
        beta_raw = np.zeros(self.M_CN)
        routing_raw = np.zeros(self.N_DN, dtype=int)

        # 这里计算的是相对大小不是精准的能耗
        H_list = []
        for i,dn in enumerate(env.DN_list):
            H_alpha = dn.body_flops * Dn_phys[i] #这里使用MFlops
            H_list.append(np.round(H_alpha,3))

        # for dn in dn_list:
        #     dn_list_id.append(dn.id)
        #     H_alpha += dn.body_flops * dn.D_req  # 这里是MFlops
        # H_alpha_list.append(H_alpha / 1e3)  # 将MFlops转化为GFlops
        # group_idx.append(dn_list_id)

        # 将物理数据块，按从大到小拿出来，填最空的桶
        sorted_dn_desc_idx = np.argsort(H_list)[::-1]
        for n in sorted_dn_desc_idx:
            lightest_cn = np.argmin(beta_raw)
            routing_raw[n] = lightest_cn
            beta_raw[lightest_cn] += H_list[n]

        # 此时 beta_raw 是分配出的 M 个"负载包裹"

        # =========================================================
        # 3. 求解算力节点合同 (CN Contract) - 包含事后修正与重匹配
        # =========================================================

        # A. 提取真实电量并转换为成本系数
        theta_raw = np.zeros(self.M_CN)
        # for i, cn in enumerate(env.CN_list):
        #     theta_raw[i] = cn.energy_remaining
        # type_list = [1.0 - (i + 1) * 0.05 for i in range(self.M_CN)]  # 随机分配类型
        # # [电量最多(最好) -> 电量最少(最差)] 的原始 ID
        # theta_desc = theta_raw[theta_sort_idx]
        theta_desc = np.zeros(self.M_CN)
        for i in range(self.M_CN):
            theta_desc[i] = 1.0 - (i + 1) * 0.05

        theta_sort_idx = np.argsort(theta_desc)[::-1]
        LAMBDA = 1.0
        # 成本系数: [小 -> 大] (Index 0 最好)
        cn_costs_asc = LAMBDA / (theta_desc + 1e-5)

        # B. 计算 CN 的虚拟成本 J_cn 并池化
        J_cn_raw = np.zeros(self.M_CN)
        cum_prob_cn = 0
        for k in range(self.M_CN):
            if k == 0:
                J_cn_raw[k] = cn_costs_asc[k]
            else:
                cost_diff = cn_costs_asc[k] - cn_costs_asc[k - 1]
                hazard_rate = cum_prob_cn / self.prob_cn[k]
                J_cn_raw[k] = cn_costs_asc[k] + hazard_rate * cost_diff
            cum_prob_cn += self.prob_cn[k]

        J_cn_ironed = self._perform_ironing(J_cn_raw)

        # C. 求解最优频率 f_m* (基于软惩罚模型的解析解)
        # f* = (eta / 2*mu*J_k)^(1/3)
        fm_star_sorted = np.zeros(self.M_CN)
        for k in range(self.M_CN):
            f_calc = (self.ETA / (2 * self.MU_COEFF * (J_cn_ironed[k] + 1e-9))) ** (1.0 / 3.0)
            f_calc = f_calc*2
            fm_star_sorted[k] = np.clip(f_calc, 1e9,1e10)

        # D. 计算这 M 个"负载包裹"的估算物理重量
        # 为了排序包裹，我们用系统平均频率来估算每个包裹的能耗
        f_avg = np.mean(fm_star_sorted)
        Q_raw_bundles = self.MU_COEFF * beta_raw * (f_avg ** 2) + self.E_COMM_UNIT * beta_raw

        # E. 【核心：事后重匹配】强制按单调性对齐
        # 把包裹从重到轻排序
        q_sort_idx = np.argsort(Q_raw_bundles)[::-1]

        # 此时，第 k 个大包裹的载荷是 beta_sorted[k]
        beta_sorted = beta_raw[q_sort_idx]

        # 重新用真实对应的最优频率，计算对齐后的精确物理成本
        Q_exact_sorted = self.MU_COEFF * beta_sorted * (fm_star_sorted ** 2) + self.E_COMM_UNIT * beta_sorted

        # F. IC 递推计算总激励 Rm_total
        Rm_total_sorted = np.zeros(self.M_CN)
        U_cn_acc = 0

        # 从最差节点 (Index M-1) 倒推
        for k in range(self.M_CN - 1, -1, -1):
            c_curr = cn_costs_asc[k]
            q_curr = Q_exact_sorted[k]

            if k == self.M_CN - 1:
                Rm_total_sorted[k] = c_curr * q_curr
            else:
                c_worse = cn_costs_asc[k + 1]
                q_worse = Q_exact_sorted[k + 1]
                rent = (c_worse - c_curr) * q_worse
                U_cn_acc += rent
                Rm_total_sorted[k] = c_curr * q_curr + U_cn_acc

        # G. 映射回物理环境
        Rm_total_final = np.zeros(self.M_CN)
        fm_final = np.zeros(self.M_CN)
        final_routing = np.zeros(self.N_DN, dtype=int)
        final_beta_m = np.zeros(self.M_CN)

        for k in range(self.M_CN):
            phys_node_id = theta_sort_idx[k]  # 应该接活的物理节点 ID
            orig_bucket_id = q_sort_idx[k]  # 原始路由生成的桶 ID

            # 把计算好的钱和频率交给这个物理节点
            Rm_total_final[phys_node_id] = Rm_total_sorted[k]
            fm_final[phys_node_id] = fm_star_sorted[k]
            final_beta_m[phys_node_id] = beta_sorted[k]

            # 修改路由表：凡是原计划去 orig_bucket_id 的数据，改道去 phys_node_id
            for n in range(self.N_DN):
                if routing_raw[n] == orig_bucket_id:
                    final_routing[n] = phys_node_id

        # =========================================================
        # 4. 带宽分配 (Bandwidth)
        # =========================================================
        W_alloc = np.ones(self.N_DN + self.M_CN) * (self.cfg.TOTAL_BW / (self.N_DN + self.M_CN))

        total_D = np.sum(Dn_optimal) + 1e-9
        for n in range(self.N_DN):
            # 将总带宽的一半按数据量比例分给 DN
            W_alloc[n] = (self.cfg.TOTAL_BW / 2.0) * (Dn_phys[n] / total_D)

        for m in range(self.M_CN):
            W_alloc[self.N_DN + m] = (self.cfg.TOTAL_BW / 2.0) / self.M_CN

        # =========================================================
        # 5. 返回动作字典
        # =========================================================
        return {
            'Dn': Dn_phys,
            'Rn': Rn_phys,
            'Rm': Rm_total_final,
            'fm': fm_final,
            'bandwidth': W_alloc,
            'routing': final_routing,
            'beta_m': final_beta_m,
            'mode': 'TCT_BASELINE'
        }

if __name__ == '__main__':
    from Contract_Env_2 import Contract_Environment
    from Contract_Config import Config
    config = Config()
    env = Contract_Environment(config)
    test = TraditionalContractBaseline(config)
    result = test.get_action(env)
    print(result['Dn'])
    state, reward, done, uav_info, dn_contract, cn_contract, uti_, total_data = env.step2(result)
    # print( reward, done, uav_info, dn_contract, cn_contract, uti_)
    print("total_data", total_data)