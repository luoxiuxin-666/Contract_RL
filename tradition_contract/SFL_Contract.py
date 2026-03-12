import numpy as np


class TraditionalContractBaseline:
    """
    传统合同理论 (Traditional Contract Theory, TCT) 求解器
    特点: 包含虚拟成本计算与强制池化(Ironing)逻辑，保证合同的严格 IC 约束。
    """

    def __init__(self, config):
        self.cfg = config
        self.N_DN = config.N_DN
        self.M_CN = config.M_CN

        # 假设类型概率是均匀分布
        self.prob_dn = np.ones(self.N_DN) / self.N_DN
        self.BETA_1 = 400.0  # 收益系数

    def _perform_ironing(self, virtual_costs):
        """
        核心池化算法 (Ironing Algorithm)
        目标: 确保虚拟成本 J_k 是严格非递减的 (即 J_0 <= J_1 <= ... <= J_{N-1})
        如果出现逆序 (如 J_k > J_{k+1})，则将它们合并，取其概率加权平均值。
        """
        N = len(virtual_costs)
        j_ironed = np.copy(virtual_costs)

        # 简单的一维池化算法 (Pool Adjacent Violators Algorithm, PAVA 的一种变体)
        # 这里因为是均匀分布，概率相等，直接取算术平均即可。
        # 不断扫描直到序列完全单调
        is_monotonic = False
        while not is_monotonic:
            is_monotonic = True
            for k in range(N - 1):
                # 如果前一个(更好的类型)的虚拟成本，竟然比后一个(更差的类型)还高
                # 破坏了单调性，必须合并
                if j_ironed[k] > j_ironed[k + 1]:
                    # 计算这两项的平均虚拟成本
                    avg_cost = (j_ironed[k] + j_ironed[k + 1]) / 2.0

                    # 强制抹平
                    j_ironed[k] = avg_cost
                    j_ironed[k + 1] = avg_cost

                    # 如果有连续三个或更多逆序，需要继续向前回溯，但这里为了简便，
                    # 设置 is_monotonic=False，外层 while 循环会重新扫描整个数组，直至完全平滑。
                    is_monotonic = False

        return j_ironed

    def get_action(self, env):
        # =========================================================
        # 1. 数据节点合同 (DN Contract) - 虚拟成本法 + 池化
        # =========================================================

        # A. 从环境获取真实成本并排序 (Index 0 最便宜，Index N-1 最贵)
        dn_costs_raw = np.zeros(self.N_DN)
        for i, dn in enumerate(env.DN_list):
            dn_costs_raw[i] = dn.unit_cost

        dn_sort_idx = np.argsort(dn_costs_raw)
        dn_costs_asc = dn_costs_raw[dn_sort_idx]

        # B. 计算原始虚拟成本 (Virtual Cost: J)
        J_raw = np.zeros(self.N_DN)
        cum_prob_better = 0

        for k in range(self.N_DN):
            if k == 0:
                J_raw[k] = dn_costs_asc[k]
                cum_prob_better += self.prob_dn[k]
            else:
                cost_diff = dn_costs_asc[k] - dn_costs_asc[k - 1]
                hazard_rate = cum_prob_better / self.prob_dn[k]
                J_raw[k] = dn_costs_asc[k] + hazard_rate * cost_diff
                cum_prob_better += self.prob_dn[k]

        # C. 【新增】执行池化 (Ironing)，消除逆序
        J_ironed = self._perform_ironing(J_raw)

        # D. 求解最优数据量 D*
        Dn_optimal = np.zeros(self.N_DN)
        for k in range(self.N_DN):
            d_star = (self.BETA_1 / (J_ironed[k] + 1e-9)) - 1
            d_star = np.clip(d_star, 0, self.cfg.max_data_count)
            Dn_optimal[k] = d_star

        # 此时由于 J_ironed 单调递增，计算出的 Dn_optimal 必然是单调递减的 (或者持平)
        # 即: Dn[0] >= Dn[1] >= ... >= Dn[N-1]

        # E. IC 递推求解激励 Rn
        Rn_optimal = np.zeros(self.N_DN)
        U_acc = 0
        for k in range(self.N_DN - 1, -1, -1):
            if k == self.N_DN - 1:
                # 差节点无租金
                Rn_optimal[k] = dn_costs_asc[k] * Dn_optimal[k]
            else:
                # 租金增量
                rent = (dn_costs_asc[k + 1] - dn_costs_asc[k]) * Dn_optimal[k + 1]
                U_acc += rent
                Rn_optimal[k] = (dn_costs_asc[k] * Dn_optimal[k]) + U_acc

        # F. 映射回物理节点顺序
        Dn_phys = np.zeros(self.N_DN)
        Rn_phys = np.zeros(self.N_DN)
        Dn_phys[dn_sort_idx] = Dn_optimal
        Rn_phys[dn_sort_idx] = Rn_optimal

        # =========================================================
        # 2. 路由决策 (Greedy Routing)
        # =========================================================
        beta_m = np.zeros(self.M_CN)
        routing = np.zeros(self.N_DN, dtype=int)

        # 将数据从大到小拿出来，填最空的桶
        # 理论上应该计算对应的能耗，但这里简化为数据量
        sorted_dn_desc = np.argsort(Dn_phys)[::-1]

        for n in sorted_dn_desc:
            lightest_cn = np.argmin(beta_m)
            routing[n] = lightest_cn
            beta_m[lightest_cn] += Dn_phys[n]

        # =========================================================
        # 3. 算力节点合同 (CN Contract) - 包含池化逻辑
        # =========================================================
        TAU_REQ = 1.0
        fm_optimal = np.zeros(self.M_CN)

        # A. 计算必需的频率
        for m in range(self.M_CN):
            if beta_m[m] > 1e-5:
                workload_flops = beta_m[m] * 1e9
                req_freq = workload_flops / TAU_REQ
                fm_optimal[m] = np.clip(req_freq, 1e9, 1e10)
            else:
                fm_optimal[m] = 1e9

        # B. 计算物理负载指标 Q
        MU_COEFF = 1e-28 * 1e9
        E_COMM_UNIT = 0.01
        Q_raw = MU_COEFF * beta_m * (fm_optimal ** 2) + E_COMM_UNIT * beta_m

        # C. 提取真实电量并转换为成本系数 (从小到大排序)
        theta_raw = np.zeros(self.M_CN)
        for i, cn in enumerate(env.CN_list):
            theta_raw[i] = cn.energy_remaining

        theta_sort_idx = np.argsort(theta_raw)[::-1]  # 电量多->少
        theta_desc = theta_raw[theta_sort_idx]

        LAMBDA = 1.0
        cn_costs_asc = LAMBDA / (theta_desc + 1e-5)  # 成本系数 小->大

        # 强制把最重的负载 Q 派给最好的节点 (即成本系数最小的节点)
        Q_sorted = np.sort(Q_raw)[::-1]  # 负载 大->小

        # D. 【新增】算力节点的池化 (Ironing)
        # 算力节点的类型分布同样可能不均匀，导致虚拟成本反常
        J_cn_raw = np.zeros(self.M_CN)
        cum_prob_cn = 0
        prob_cn_k = 1.0 / self.M_CN

        for k in range(self.M_CN):
            if k == 0:
                J_cn_raw[k] = cn_costs_asc[k]
                cum_prob_cn += prob_cn_k
            else:
                cost_diff = cn_costs_asc[k] - cn_costs_asc[k - 1]
                hazard_rate = cum_prob_cn / prob_cn_k
                J_cn_raw[k] = cn_costs_asc[k] + hazard_rate * cost_diff
                cum_prob_cn += prob_cn_k

        # 执行池化，确保 J_cn_ironed 严格递增
        J_cn_ironed = self._perform_ironing(J_cn_raw)

        # 重新修正负载分配 Q (如果 J 被合并了，意味着这两个类型的节点应该接一样的活)
        # 注意: 因为路由是由 DN 贪婪决定的，Q_sorted 已经被物理上固定了
        # TCT 面临的困境: 我们无法随便改 Q_sorted。如果此时 J 被 Ironing 了，
        # 在严格的数学意义上，我们需要回去改路由，让他们的 beta 一样。
        # 为了工程实现，我们近似认为：对被 Ironing 绑定的节点，强行求 Q 的平均值。
        Q_ironed = np.copy(Q_sorted)
        for k in range(self.M_CN - 1):
            if J_cn_ironed[k] == J_cn_ironed[k + 1]:
                avg_q = (Q_ironed[k] + Q_ironed[k + 1]) / 2.0
                Q_ironed[k] = avg_q
                Q_ironed[k + 1] = avg_q

        # E. IC 递推计算算力激励 Rm (使用修正后的 Q_ironed)
        Rm_total_sorted = np.zeros(self.M_CN)
        U_cn_acc = 0

        for k in range(self.M_CN - 1, -1, -1):
            c_curr = cn_costs_asc[k]
            q_curr = Q_ironed[k]

            if k == self.M_CN - 1:
                Rm_total_sorted[k] = c_curr * q_curr
            else:
                c_worse = cn_costs_asc[k + 1]
                q_worse = Q_ironed[k + 1]
                rent = (c_worse - c_curr) * q_worse
                U_cn_acc += rent
                Rm_total_sorted[k] = (c_curr * q_curr) + U_cn_acc

        # F. 映射回物理顺序
        Rm_total_final = np.zeros(self.M_CN)
        Rm_total_final[theta_sort_idx] = Rm_total_sorted

        # =========================================================
        # 4. 带宽分配 & 返回
        # =========================================================
        W_alloc = np.ones(self.N_DN + self.M_CN) * (self.cfg.TOTAL_BW / (self.N_DN + self.M_CN))

        return {
            'Dn': Dn_phys, 'Rn': Rn_phys,
            'Rm_total': Rm_total_final, 'fm': fm_optimal,
            'bandwidth': W_alloc, 'routing': routing,
            'beta_m': beta_m, 'mode': 'TCT_BASELINE'
        }