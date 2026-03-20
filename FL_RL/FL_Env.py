import numpy as np
import copy


# ==========================================
# 1. 数据节点 (Data Node) - FL Mode
# ==========================================
class DataNode:
    def __init__(self, config_DN, id=None):
        self.id = id
        self.max_data_count = config_DN.max_data_count

        # 物理硬件
        self.f_n = np.random.uniform(config_DN.f_n[0], config_DN.f_n[1])  # 本地计算频率
        self.mu_n = config_DN.mu_dn
        self.kappa = config_DN.kappa

        # 通信参数
        self.P_tx = config_DN.P_UAVFDN  # 上传功率
        self.SINR = 0.0
        self.bandwidth = 0.0

        # 模型参数
        # FL 需要跑完整模型，假设 Full FLOPs 是 Head 的 5~10 倍
        self.unit_full_flops = config_DN.UNIT_FULL_FLOPS  # GFLOPs/Sample
        self.model_size_mbits = config_DN.MODEL_SIZE_MBITS  # 完整模型大小

        # 成本与效用
        self.unit_cost = 0.0  # 这里的 unit_cost 是综合系数
        self.unit_time = 0.0  # 单位时间
        self.D_req = 0
        self.R_offer = 0
        self.utility = 0.0
        self.T = config_DN.T

        # 计算自身的单位能耗系数 (Type)
        self.total_energy = 0.0
        self.total_time = 0.0
        # Cost = c * D
        # 估算单位计算能耗 + 单位通信能耗(假设平均信道)
        self._estimate_unit_cost()

    def _estimate_unit_cost(self):
        unit_full_total = self.unit_full_flops  # 单位数据所需要的完整浮点运算：MFLOPs
        speed = (self.f_n * self.kappa) / 1e6  # 运算速度:单位 MFlops/s
        p_n = self.mu_n * (self.kappa*self.f_n ** 3.0)  # 运算功率

        full_time_s = unit_full_total / speed if speed > 0 else 0  # 单位计算时间
        full_energy = p_n * full_time_s  # 单位计算能耗

        self.unit_time = np.round(full_time_s,4)
        self.unit_cost = np.round(full_energy,3)

    def calculate_real_cost(self, D_req, bandwidth, flag=False):
        """
        计算真实的物理成本 (Energy)
        """
        # 1. 计算能耗
        e_cmp = D_req * self.unit_cost * self.T
        t_cmp = D_req * self.unit_time * self.T

        # 2. 通信能耗 (上传模型)
        # Rate = B * log2(1 + SINR)
        rate = (bandwidth * np.log2(1 + self.SINR)) / 1e6
        t_trans = self.model_size_mbits / (rate + 1e-9)
        e_com = self.P_tx * t_trans

        total_energy = e_cmp + e_com*2

        # 3. 计算时间 (用于 UAV 惩罚)
        total_time = t_cmp + t_trans*2

        if flag:
            self.total_energy = total_energy
            self.total_time = total_time

        return total_energy, total_time

    def evaluate_contract(self, D_req, R_offer, bandwidth,flag = False):
        """
        IR 检查
        """
        # 如果数据量超过物理上限，成本无穷大
        if D_req > self.max_data_count:
            return -1e9

        real_cost, _ = self.calculate_real_cost(D_req, bandwidth,flag)

        # 归一化成本 (为了数值稳定)
        # scaled_cost = real_cost * 1e-4
        scaled_cost = real_cost

        self.utility = R_offer - scaled_cost
        return self.utility

    def reset(self):
        self.D_req = 0
        self.R_offer = 0


# ==========================================
# 2. 无人机 (UAV)
# ==========================================
class FL_UAV:
    def __init__(self, config):
        self.total_bw = config.TOTAL_BW
        # self.beta_1 = config.beta_1  # 收益系数
        # self.beta_2 = config.beta_2  # 成本系数
        self.N_DN = config.N_DN

        self.E_h = config.E_h
        self.total_cost = 0.0
        self.utility = 0.0
        self.T = config.T

        self.max_time = 0.0

    def calculate_utility(self, dn_list, alpha):
        total_payment = 0
        max_time = 0
        uti = 0.0
        for dn in dn_list:
            if dn.utility >= 0:  # 只有接受的节点才算
                Dn = dn.D_req / 550   # 缩小一下区间
                gain = np.log(1 + Dn)
                pay = dn.R_offer / 700
                uti += (1 / self.N_DN) * alpha * (gain - pay)
                max_time = max(max_time, dn.total_time)

        # 成本: 支付 + 时延惩罚
        # 注意: FL 的时延通常比 SFL 长很多
        # cost = self.E_h * max_time
        cost = self.E_h

        self.max_time = 0.0
        self.total_cost = 0.0
        self.max_time = max_time
        self.total_cost = cost

        self.utility = uti - cost
        return self.utility


# ==========================================
# 3. FL 环境类
# ==========================================
class FLEnvironment:
    def __init__(self, config):
        self.cfg = config
        self.N_DN = config.N_DN

        # 状态维度: [IR_State(N), SINR(N)]
        self.state_dim = self.N_DN * 2
        self.alpha_ppo = config.alpha_fl_ppo
        self.alpha_contract = config.alpha_fl_contract

        self.TOTAL_BW = config.TOTAL_BW

        self.UAV2DNSINR = 0.0

        # 动作维度由 PPO 决定 (Dn, W)

        self.DN_list = []
        self.init_nodes()
        self.uav = FL_UAV(config)

        # 状态容器
        self.sinr_list = np.zeros(self.N_DN)
        self.dn_ir = np.zeros(self.N_DN)

    def init_nodes(self):
        for i in range(self.N_DN):
            dn = DataNode(self.cfg, id=i)
            self.DN_list.append(dn)

        # 按成本系数排序 (为了合同理论)
        self.DN_list.sort(key=lambda x: x.unit_cost)  # 0: Worst, N-1: Best
        unit_cost = []
        unit_time = []
        # 2. 遍历排序后的列表，重新赋予 ID
        for new_id, dn in enumerate(self.DN_list):
            dn.id = new_id
            unit_cost.append(dn.unit_cost)
            unit_time.append(dn.unit_time)

        print("unit_cost:", unit_cost)
        print("unit_time:", unit_time)

    def init_all_SINR(self, flag=False):
        if flag:
            np.random.seed(42)
        # 无人机的信噪比
        self.UAV2DNSINR = np.random.normal(1.4, 0.4, size=1)  # 生成一个符合均值为1，标准差为0.4的随机数
        self.UAV2DNSINR = np.clip(self.UAV2DNSINR, 0.8, 2)  #

        self.sinr_list = np.random.normal(1.4, 0.4, size=self.N_DN)  # 生成verifier_nums-1个，均值为1.4，标准差为0.4的随机数
        self.sinr_list = np.clip(self.sinr_list, 0.8, 2)

    def reset(self):
        # 1. 随机生成信道
        self.init_all_SINR(False)

        # 2. 分配信道给节点 (按 ID 顺序，或者按排序后的顺序)
        # 这里假设 sinr 是随机分布在区域内的，直接按列表顺序赋值即可
        for i, dn in enumerate(self.DN_list):
            dn.SINR = self.sinr_list[i]
            dn.reset()


        self.dn_ir = np.random.randint(0, 2, self.N_DN)

        return np.concatenate([self.dn_ir, self.sinr_list])

    def reset2(self):
        self.reset()
        # 加一个随机能耗扰动
        # 基准值
        base = self.DN_list[0].unit_cost

        # 波动范围（比如 10%）
        delta = base * 0.05

        # 最终结果：base ± delta 之间随机
        result = np.random.uniform(-delta, delta)

        for dn in self.DN_list:
            dn.unit_cost += result

    def decode_action(self, proc_cont):
        """
        FL 合同生成: Dn -> Rn
        """
        N = self.N_DN

        dn_ascending = proc_cont[0:N]
        Dn_phys = dn_ascending[::-1] * self.cfg.max_data_count
        Dn_phys = np.round(Dn_phys, 0)

        # 2. 解析带宽
        bw_ratios = proc_cont[N:]
        W_phys = bw_ratios * self.cfg.TOTAL_BW

        Rn_phys = self.get_Rn_by_Dn(Dn_phys, N, W_phys)

        # 此时 Dn_phys 和 Rn_phys 都是按 [Worst -> Best] 排序的
        # 刚好对应 self.DN_list 的顺序

        return {
            'Dn': Dn_phys,
            'Rn': Rn_phys,
            'bandwidth': W_phys
        }

    def get_Rn_by_Dn(self, Dn_phys, N, W_phys):
        Rn_phys = np.zeros(N)
        Uti_Dn = np.zeros(N)
        u_acc = 0
        for k in range(N - 1, -1, -1):
            dn = self.DN_list[k]
            w = W_phys[k]

            # 根据当前合同的数据量去计算对应的能耗
            cost, _ = dn.calculate_real_cost(Dn_phys[k], w, flag=True)
            if k == N - 1:
                Uti_Dn[k] = 0
                Rn_phys[k] = cost
            else:
                # IC: 防止好人 (k) 模仿 坏人 (k-1)
                # 这里的逻辑是：k 比 k-1 成本低。
                # 租金 = (c_{k-1} - c_k) * D_{k-1}
                # 获取下一级的合同的类型
                dn_ic = self.DN_list[k + 1]
                dn_copy = copy.deepcopy(dn_ic)
                dn_copy.unit_cost = dn.unit_cost

                w_ic = W_phys[k + 1]
                data_ic = Dn_phys[k + 1]

                cost_worse, _ = dn_ic.calculate_real_cost(data_ic, w_ic)
                # 当前节点获取下一个节点的合同的能耗
                cost_curr_mimic, _ = dn_copy.calculate_real_cost(data_ic, w_ic)

                rent = cost_worse - cost_curr_mimic
                if rent < 0:
                    print("DN is error!!!")

                # 当前好人的总效用 = 差人的效用 + 新增租金
                Uti_Dn[k] = Uti_Dn[k + 1] + rent

                # 最终激励 = 物理成本 + 净效用(租金)
                Rn_phys[k] = cost + Uti_Dn[k]
        return Rn_phys

    def step(self, proc_cont):
        # 1. 生成合同
        action_dict = self.decode_action(proc_cont)
        Dn = action_dict['Dn']
        Rn = action_dict['Rn']
        W = action_dict['bandwidth']

        for i, dn in enumerate(self.DN_list):
            dn.D_req = Dn[i]
            dn.R_offer = Rn[i]

        # 2. 节点决策
        for i, dn in enumerate(self.DN_list):
            # 将生成的带宽分配给节点 (这里可能需要重新匹配顺序，为了简单直接按排序后的索引给)
            # 假设 W 也是 PPO 输出的对应排序后的权重
            dn.bandwidth = W[i]

            uti = dn.evaluate_contract(Dn[i], Rn[i], W[i])

            if uti >= 0:
                self.dn_ir[i] = 1
            else:
                self.dn_ir[i] = 0

        # 3. 计算系统奖励
        reward = self.uav.calculate_utility(self.DN_list,self.alpha_ppo)

        # 状态更新
        next_state = np.concatenate([self.dn_ir, self.sinr_list])

        contract = np.concatenate([Dn, Rn])

        total_data = sum(Dn)

        return next_state, reward, False,contract, total_data

    def step2(self,contract):
        Dn = contract['Dn']
        W = contract['bandwidth']

        Rn = self.get_Rn_by_Dn(Dn,self.N_DN, W)
        contract['Rn'] = Rn

        for i, dn in enumerate(self.DN_list):
            dn.D_req = Dn[i]
            dn.R_offer = Rn[i]

        # 2. 节点决策
        for i, dn in enumerate(self.DN_list):
            # 将生成的带宽分配给节点 (这里可能需要重新匹配顺序，为了简单直接按排序后的索引给)
            # 假设 W 也是 PPO 输出的对应排序后的权重
            dn.bandwidth = W[i]

            uti = dn.evaluate_contract(Dn[i], Rn[i], W[i])

            if uti >= 0:
                self.dn_ir[i] = 1
            else:
                self.dn_ir[i] = 0

        # 3. 计算系统奖励
        reward = self.uav.calculate_utility(self.DN_list,self.alpha_contract)

        # 状态更新
        next_state = np.concatenate([self.dn_ir, self.sinr_list])

        contract = np.concatenate([Dn, Rn])

        total_data = sum(Dn)

        return next_state, reward, False, contract, total_data

    def step3(self,action):
        '''
        return {
        'Dn': Dn_phys,
        'Rn': Rn_phys,
        'bandwidth': W_alloc,
        'mode': 'FL_PRICING_BASELINE'  # 让环境 step 知道此时是 FL 模式，计算效用不要带 CN
        }
        '''

        Dn = action['Dn']
        Rn = action['Rn']
        W = action['bandwidth']

        for i, dn in enumerate(self.DN_list):
            dn.D_req = Dn[i]
            dn.R_offer = Rn[i]

        # 2. 节点决策
        for i, dn in enumerate(self.DN_list):
            # 将生成的带宽分配给节点 (这里可能需要重新匹配顺序，为了简单直接按排序后的索引给)
            # 假设 W 也是 PPO 输出的对应排序后的权重
            dn.bandwidth = W[i]

            uti = dn.evaluate_contract(Dn[i], Rn[i], W[i],flag = True)

            if uti >= 0:
                self.dn_ir[i] = 1
            else:
                self.dn_ir[i] = 0

        # 3. 计算系统奖励
        reward = self.uav.calculate_utility(self.DN_list, self.alpha_ppo)

        # 状态更新
        next_state = np.concatenate([self.dn_ir, self.sinr_list])

        contract = np.concatenate([Dn, Rn])

        total_data = sum(Dn)

        return next_state, reward, False, contract, total_data

