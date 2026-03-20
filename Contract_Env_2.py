import copy

import numpy as np
# 设置随机种子
from FSL_Model import SFL_CNN, get_sfl_metrics, model_size_by_sfl

rng = np.random.default_rng(seed=42)

DN_Time = []
CN_Time = []


def generate_balanced_distribution(N, distinctness=20):
    """
    N: 对象个数
    distinctness: 集中度参数 (alpha)。
                  数值越大，结果越平均（越接近 1/N）；
                  数值越小，差距越大。
                  建议取值 10 ~ 100 之间满足"不要相差太远"的需求。
    """
    # 构造 alpha 参数列表，所有元素相同表示每个人权重预期一样
    alpha = np.ones(N) * distinctness

    # 生成分布
    distribution = rng.dirichlet(alpha)

    return distribution


# ==========================================
# 1. 数据节点 (Data Node)
# ==========================================
class DataNode:
    def __init__(self, config_DN, L_c, id=None, location=None):
        self.id = id
        self.max_data_size = config_DN.max_data_size  # 物理最大数据拥有量
        self.location = location if location else np.random.rand(2) * 100

        # 动态状态
        self.current_data_size = 0  # 本轮要训练的数据量 (Dn) 单位MB
        self.max_data_count = config_DN.max_data_count
        self.MB_unit_ = config_DN.MB_unit_  # 单位MB的bit
        # 单张图片所需要的能耗
        self.unit_cost = 0.0
        # 单张图片所需要的时间
        self.unit_time = 0.0
        self.other_energy = 0
        self.other_time = 0.0
        self.Request_Data = 0
        self.D_req = 0
        self.R_offer = 0.0
        self.T = config_DN.T

        # 物理硬件条件
        self.f_n = rng.integers(config_DN.f_n[0], config_DN.f_n[1])  # cpu处理能力从范围内随机选取
        self.kappa = config_DN.kappa  # 16 FLOPs/cycle
        # 在 Contract_Env.py 第44行之前添加：
        # self.mu_n = rng.uniform(config_DN.mu_dn[0], config_DN.mu_dn[1])
        self.mu_n = config_DN.mu_dn
        # 算力节点的通信条件
        self.P_to_CN = rng.uniform(config_DN.DN_P_1, config_DN.DN_P_2)
        self.P_from_CN = rng.uniform(config_DN.DN_P_1, config_DN.DN_P_2)

        # 与无人机的通信条件
        self.P_to_UAV = rng.uniform(config_DN.DN_P_1, config_DN.DN_P_2)
        self.P_from_UAV = rng.uniform(config_DN.DN_P_1, config_DN.DN_P_2)

        # 对于算力节点的信噪比,初始化为0，后面再重新赋值
        self.DN2CN_SINR = 0.0
        self.DN2CN_W = 0.0

        self.DN2UAV_SINR = 0.0
        self.DN2UAV_W = 0.0  # 这一部分需要无人机分配，初始化的话就随机

        self.CN2DN_SINR = 0.0
        self.CN2DN_W = 0.0

        self.UAV2DN_SINR = 0.0
        self.UAV2DN_W = 0.0

        # self.model_param = config_DN.model_param    #模型参数

        self.L_t = config_DN.L_t  # 固定将最后一层
        self.L_c = L_c  # 随机分割层
        # 尾部网络的计算量
        self.tail_cmp_energy = 0.0
        self.tail_flops = 0.0
        # 尾部网络的粉碎数据大小
        self.tail_com_data = 0.0
        # 计算客户端模型参数单位Mbit
        self.client_model_size, self.total_model_size = model_size_by_sfl(self.L_c)

        self.tail_model_size = 0.0
        self.compare_tail_()

        # 计算卸载出去的任务大小
        self.body_flops = 0
        self.body_model_size = self.total_model_size - self.client_model_size - self.tail_model_size
        self.CN2DN_Com_Time_Unit = 0.0
        self.DN2CN_Com_Time_Unit = 0.0
        self.DN2UAV_Com_Time_Pram = 0.0
        self.DN2CN_Com_Time_Pram = 0.0

        # 无人机时间
        self.UAV2DN_Com_Time = 0.0
        self.DN2UAV_Com_Time = 0.0

        self.cmp_time = 0.0

        # 总的粉碎数据和梯度数据（单向）
        self.com_data = 0.0

        self.energy_cost = 0.0

    def compare_tail_(self):
        ret_tail = get_sfl_metrics(self.L_t, self.f_n, self.kappa, self.mu_n, 1)
        self.tail_flops = ret_tail["Full_Total_MFLOPs"] - ret_tail["Client_Total_MFLOPs"]
        self.tail_cmp_energy = ret_tail["Total_Energy_J"] - ret_tail["Batch_Energy_J"]
        self.tail_com_data = ret_tail["Unit_Data_Mbits"]
        client_model_size, total_model_size = model_size_by_sfl(self.L_t)
        self.tail_model_size = total_model_size - client_model_size

    def evaluate_contract(self, DN2UAV_W, D_req, R_offer=None, Flag=False):
        """
        评估数据合同 (Dn, Rn)
        IR 约束: 激励 >= 成本
        """
        other_energy, other_time = self.compute_other_energy(DN2UAV_W, Flag)
        # 简单成本模型: 数据训练成本 + 通信成本
        # self.current_data_size = D_req
        energy_cost = self.T * self.unit_cost * D_req + other_energy
        # energy_cost = energy_cost * 1e-3 #将单位转化为 KJ
        if R_offer is None:
            return energy_cost

        self.energy_cost = energy_cost
        self.other_energy = other_energy
        self.other_time = other_time

        self.D_req = D_req
        self.R_offer = R_offer

        utility = R_offer - self.energy_cost
        return utility

    def compute_unit_cost(self):
        # 计算分割层所需要的浮点运算数
        """
        :return:
        "Lc": cut_layer,
        # 1. 完整模型指标 (常数)
        "Full_Total_MFLOPs": unit_full_total / 1e6,

        # 2. 客户端指标
        "Client_Total_MFLOPs": unit_client_total / 1e6,
        "Batch_Time_ms": batch_time_s * 1000,
        "Batch_Energy_J": batch_energy_j,

        # 3. 服务器指标
        "Server_Total_MFLOPs": unit_server_total / 1e6,

        # 4. 通信指标
        "Unit_Data_Mbits": unit_data_mbits
        """
        # TODO对齐一下单位
        ret_1 = get_sfl_metrics(self.L_c, self.f_n, self.kappa, self.mu_n, 1)  # 计算的时候使用单位数据
        client_Mflops = ret_1["Client_Total_MFLOPs"]
        total_Mflops = ret_1["Full_Total_MFLOPs"]
        client_energy = ret_1['Batch_Energy_J']
        client_data_mbit = ret_1['Unit_Data_Mbits']
        dn2cn_r = (self.DN2CN_W * np.log2(1 + self.DN2CN_SINR)) / 1e6  # Mbps
        cn2dn_r = (self.CN2DN_W * np.log2(1 + self.CN2DN_SINR)) / 1e6
        speed = (self.f_n * self.kappa) / 1e6
        self.body_flops = total_Mflops - client_Mflops - self.tail_flops
        cmp_Mflops = client_Mflops + self.tail_flops
        self.cmp_time = cmp_Mflops / speed  # 单位s

        comp_energy = client_energy + self.tail_cmp_energy

        # 计算通信能耗
        self.com_data = client_data_mbit + self.tail_com_data  # 单位通信参数大小
        # 对齐单位  Mbits/Mbps
        # 带宽使用Hz，
        self.DN2CN_Com_Time_Unit = self.com_data / (dn2cn_r * 2)

        self.CN2DN_Com_Time_Unit = self.com_data / (cn2dn_r * 2)

        com_energy = self.P_to_CN * self.DN2CN_Com_Time_Unit + self.P_from_CN * self.CN2DN_Com_Time_Unit

        self.unit_cost = com_energy + comp_energy
        self.unit_cost = np.round(self.unit_cost, 3)

        self.unit_time = self.cmp_time + self.DN2CN_Com_Time_Unit + self.CN2DN_Com_Time_Unit
        self.unit_time = np.round(self.unit_time, 4)

    def compute_other_energy(self, DN2UAV_W, flag=False):
        # 接收无人机的模型大小
        uav2dn_r = (self.UAV2DN_W * np.log2(1 + self.UAV2DN_SINR)) / 1e6  # Mbps
        dn2uav_r = (DN2UAV_W * np.log2(1 + self.DN2UAV_SINR)) / 1e6
        dn2cn_r = (self.DN2CN_W * np.log2(1 + self.DN2CN_SINR)) / 1e6
        # 对齐一下单位
        UAV2DN_Com_Time = self.total_model_size / uav2dn_r

        DN2UAV_Com_Time = (self.client_model_size + self.tail_model_size) / dn2uav_r

        DN2CN_Com_Time_Pram = self.body_model_size / dn2cn_r

        other_energy = self.P_from_UAV * UAV2DN_Com_Time + self.P_to_UAV * DN2UAV_Com_Time + self.P_to_CN * DN2CN_Com_Time_Pram

        other_time = UAV2DN_Com_Time + DN2UAV_Com_Time + DN2CN_Com_Time_Pram
        # if other_time > 600:
        #     print('dn_other_time > 600:', other_time)
        if flag:
            self.UAV2DN_Com_Time = UAV2DN_Com_Time
            self.DN2UAV_Com_Time = DN2UAV_Com_Time
            self.DN2CN_Com_Time_Pram = DN2CN_Com_Time_Pram
        return other_energy, other_time

    def reset(self):
        self.D_req = 0.0
        self.R_offer = 0.0
        self.energy_cost = 0.0
        self.DN2UAV_W = 0.0
        self.other_energy = 0.0
        # self.total_time = 0.0


# ==========================================
# 2. 算力节点 (Compute Node)
# ==========================================
class ComputeNode:
    def __init__(self, config_CN, max_freq, energy_remaining, type, id=None, location=None):
        self.id = id
        self.max_freq = max_freq  # 物理最大频率 (fm_max)
        # self.total_energy = config_CN.total_energy  # 电池总容量
        self.location = location if location else np.random.rand(2) * 100
        self.fm = 0

        # 物理常数 (根据论文设定)
        self.mu = 1e-28  # 能耗系数 alpha
        self.lam = 1.0  # 效用转化系数

        self.kappa = config_CN.kappa  # 16 FLOPs/cycle

        # 动态状态
        self.energy_remaining = energy_remaining  # 绝对剩余能量 (theta_m)

        self.type = type
        self.is_active = True  # 是否因没电掉线

        self.DN_list = None  # 获取所有的数据节点的信息
        self.H_alpha = 0.0
        self.cmp_time = 0.0

        # 对于数据节点的信噪比
        self.DN2CN_SINR = 0.0
        self.DN2CN_W = 0.0

        self.CN2UAV_SINR = 0.0
        self.CN2UAV_W = 0.0  # 这一部分需要无人机分配，初始化的话就随机

        self.CN2DN_SINR = 0.0
        self.Total_W = 0.0

        self.UAV2CN_SINR = 0.0
        self.UAV2CN_W = 0.0

        self.P_to_DN = config_CN.P_CN2DN
        self.P_from_DN = config_CN.P_CNFDN

        self.P_to_UAV = config_CN.P_CN2UAV
        self.P_from_UAV = config_CN.P_CNFUAV
        self.T = config_CN.T

        # 通信时间和能耗
        self.E_trans = 0.0
        self.total_com_time = 0.0
        self.body_model_size = 0

        self.CN2UAV_com_time = 0.0
        self.Other_COM_E = 0.0

        self.total_E = 0.0
        self.total_time = 0.0

        self.R_offer = 0.0

        self.max_total_time = 0.0

    def calculate_energy_by_contract(self, CN2UAV_W, DN_list, f_m_prob, flag=False):
        """
        根据合同计算对应的能耗
        按照合同的定义，其接收对应id的数据节点卸载过来的任务
        """
        # CN2DN_W = self.Total_W / len(DN_list) #将带宽平均分给所有的数据节点
        if f_m_prob <= 1:
            f_m = f_m_prob * self.max_freq
        else:
            f_m = f_m_prob
        p_m = float(self.mu) * float(self.kappa) * (float(f_m) ** 3.0) / 2
        speed = f_m * self.kappa / 1e9
        H_alpha = 0.0
        body_model_size = 0.0
        E_trans = 0.0
        total_com_time = 0.0
        current_time = 0.0
        for dn in DN_list:
            H_alpha += (dn.body_flops * dn.D_req) / 1e3  # 将FLOPS转化为GFlops
            E_trans += (self.P_from_DN * dn.DN2CN_Com_Time_Unit + self.P_to_DN * dn.CN2DN_Com_Time_Unit) * dn.D_req
            current_time = (dn.DN2CN_Com_Time_Unit + dn.CN2DN_Com_Time_Unit) * dn.D_req
            total_com_time = max(current_time, total_com_time)
            body_model_size += dn.body_model_size

        cmp_time = H_alpha / speed
        cmp_energy = cmp_time * p_m

        CN2UAV_com_time = body_model_size / (CN2UAV_W * np.log2(1 + self.CN2UAV_SINR))
        Other_COM_E = CN2UAV_com_time * self.P_to_UAV

        total_E = self.T * (cmp_energy + E_trans) + Other_COM_E
        total_time = self.T * (cmp_time + total_com_time) + CN2UAV_com_time

        if flag:
            self.H_alpha = H_alpha
            self.E_trans = E_trans
            self.total_com_time = total_com_time
            self.body_model_size = body_model_size
            self.cmp_time = cmp_time
            self.CN2UAV_com_time = CN2UAV_com_time
            self.Other_COM_E = Other_COM_E
            self.total_time = total_time

            if total_time > 1000 and total_time > self.max_total_time:
                self.max_total_time = total_time
                print(f" total_time is {self.max_total_time}")

        return total_E, total_time, H_alpha

    def evaluate_contract(self, CN2UAV_W, DN_list, f_m, flag=False, R_offer=None):
        """
        评估数据合同 (Dn, Rn)
        IR 约束: 激励 >= 成本
        """
        total_E, total_time, H_alpha = self.calculate_energy_by_contract(CN2UAV_W, DN_list, f_m, flag)
        # total_E = total_E / 100     # 将数值限制在某个范围
        if flag:
            self.total_E = total_E
        if R_offer is None:
            return (1 / self.type) * total_E, H_alpha

        # 简单成本模型: 数据训练成本 + 通信成本
        if flag:
            self.R_offer = R_offer
            self.total_E = total_E
            self.total_time = total_time

        utility = R_offer - (1 / self.type) * total_E

        return utility

    def reset(self):
        self.fm = 0
        self.H_alpha = 0.0
        self.cmp_time = 0.0
        # 这里只重置信道等
        self.E_trans = 0.0
        self.total_com_time = 0.0
        self.body_model_size = 0

        self.CN2UAV_com_time = 0.0
        self.Other_COM_E = 0.0

        self.total_E = 0.0
        self.total_time = 0.0

        self.R_offer = 0.0

        self.max_total_time = 0.0


# ==========================================
# 3. 无人机/服务器 (UAV / Agent)
# ==========================================
class UAV:
    def __init__(self, config_uav, location=None):
        self.cfg = config_uav
        self.total_bandwidth = config_uav.TOTAL_BW  # W_total
        self.location = location if location else np.array([50, 50, 100])  # 高度100
        self.total_data = config_uav.max_data_size

        self.beta_1 = 0
        self.beta_2 = config_uav.beta_2
        self.mode = 'ppo'
        self.alpha_dn = 0

        self.DN_list = None
        self.CN_list = None

        self.max_dn_uti = 0.0
        self.max_cn_uti = 0.0
        self.data = [0.0, 0.0]

        # 物理能耗硬件
        self.f_p = config_uav.f_p
        self.E_h = config_uav.E_h  # 单位悬停能耗
        self.kappa = config_uav.kappa

        self.mu = config_uav.mu_uav

        self.P_to_DN = config_uav.P_UAV2DN
        self.P_from_DN = config_uav.P_UAVFDN

        self.P_from_CN = config_uav.P_UAVFCN

        self.total_time = 0.0
        self.total_com_energy = 0
        self.total_cost = 0

        self.utility = 0.0

        self.max_total_cost = 0.0

    def calculate_cost(self, DN_list, CN_list):
        self.total_cost = 0.0
        self.total_time = 0.0
        self.total_com_energy = 0.0
        self.total_time = 0.0
        if not DN_list or not CN_list:
            return 0
        total_dn_time = 0.0
        for dn in DN_list:
            self.total_com_energy += self.P_to_DN * dn.UAV2DN_Com_Time + self.P_from_DN * dn.DN2UAV_Com_Time
            total_dn_time = max(dn.T * dn.cmp_time * dn.D_req + dn.other_time, total_dn_time)
        total_cn_time = 0.0
        for cn in CN_list:
            self.total_com_energy += self.P_from_CN * cn.CN2UAV_com_time
            total_cn_time = max(cn.total_time, total_cn_time)

        self.total_time = total_dn_time + total_cn_time
        agg_cost = self.mu * (self.f_p * self.kappa) ** 2 * 1e6
        self.total_cost = self.total_com_energy + agg_cost + self.E_h * self.total_time
        # if self.total_cost > self.max_total_cost:
        #     self.max_total_cost = self.total_cost
        #     print(f"the max total cost is {self.max_total_cost}")
        uav_info = {
            'total_time': self.total_time,
            'total_cost': self.total_cost,
            'dn_time': total_dn_time,
            'cn_time': total_cn_time
        }
        return uav_info

    def DN_uti(self, dn):
        Dn = dn.D_req / 500  # 缩小一下区间
        uti = np.log(1 + Dn)
        r = dn.R_offer / 700
        uti = uti - r
        uti = self.alpha_dn * uti
        # if uti < 0:
        #     print("DN", uti)
            # print("DN_all_action",all_action)
        # if uti > self.max_dn_uti:
        #     self.max_dn_uti = uti
        #     print("max_dn_uti",self.max_dn_uti)
        return uti

    def CN_uti(self, cn):
        H_alpha = cn.H_alpha
        if cn.fm <= 1:
            fm = (cn.kappa * cn.fm * cn.max_freq) / 1e9
        else:
            fm = (cn.kappa * cn.fm) / 1e9
        log_1 = np.log(1 + H_alpha / 1000)
        log_2 = (H_alpha / fm)
        A = self.beta_1 * log_1
        B = self.beta_2 * log_2
        U_p = A - B
        # U_p = A
        r = cn.R_offer / 100
        Uti = U_p - r
        # if Uti < 0:
        #     print("CN", Uti)
        # if log_2 > self.data[1]:
        #     self.data[1] = log_2
        #     print("max", self.data[1])
        # if log_2 < self.data[0]:
        #     self.data[0] = log_2
        #     print("min", self.data[0])
        return Uti

    def reset(self):
        self.total_time = 0.0
        self.total_com_energy = 0
        self.total_cost = 0
        self.utility = 0.0
        self.max_total_cost = 0.0
        self.max_dn_uti = 0.0
        self.max_cn_uti = 0.0

        if self.mode == 'ppo':
            self.alpha_dn = self.cfg.alpha_sfl_ppo
            self.beta_1 = self.cfg.beta_sfl_ppo
        elif self.mode == 'contract':
            self.alpha_dn = self.cfg.alpha_sfl_contract
            self.beta_1 = self.cfg.beta_sfl_contract
        elif self.mode == 'pricing':
            self.alpha_dn = self.cfg.alpha_pricing
            self.beta_1 = self.cfg.beta_pricing


class Contract_Environment():
    def __init__(self, config):
        self.cfg = config

        # self.total_reward = self.cfg.total_reward
        # --- 系统规模 ---
        self.N_DN = self.cfg.N_DN  # 数据节点数
        self.M_CN = self.cfg.M_CN  # 算力节点数

        self.DN_prob = 1 / self.N_DN
        self.CN_prob = 1 / self.M_CN

        self.state_dim = (self.N_DN + self.M_CN) * 2
        self.action_dim = self.N_DN * 2 + self.M_CN
        # --- 物理约束 (映射常数) ---
        # self.MAX_REWARD_DN = self.cfg.MAX_REWARD_DN
        # self.MAX_REWARD = self.cfg.MAX_REWARD  # MB (Dn)
        # self.MAX_REWARD_CN = self.cfg.MAX_REWARD_CN
        # self.CN_Fm = self.cfg.CN_Fm  # GHz (fm)

        self.MAX_DATA_SIZE = self.cfg.max_data_size  # 这里修改为MB单位
        self.MAX_DATA_COUNT = self.cfg.max_data_count
        self.TOTAL_BW = self.cfg.TOTAL_BW  # MHz (Bandwidth)
        # self.tau_dn = self.cfg.tau_dn
        # self.tau_cn = self.cfg.tau_cn
        self.tau_dn = 1 / self.N_DN
        self.tau_cn = 1 / self.M_CN
        self.max_freq = self.cfg.max_freq
        # 信噪比
        self.UAV2DNSINR = None
        self.UAV2CNSINR = None
        self.DN2UAVSINR = None
        self.DN2UAVSINR = None
        self.DN2CNSINR = None
        self.CN2DNSINR = None
        self.base_channel_quality = 0.0

        self.DN_list = []
        self.CN_list = []
        self.uav = None
        self.base_freq = 1e10
        self.sort_idx = []

        self.init_all_SINR()
        self.init_List()
        self.init_UAV()

        self.dn_ir = None
        self.dn_ic = None
        self.cn_ir = None
        self.cn_ic = None

        # 状态空间
        self.state = []
        # 动作空间
        self.action = []
        # 奖励
        self.reward = 0.0

    def init_List(self):
        # 初始情况下将无人机的带宽随机分配给N+M个节点
        distributed = generate_balanced_distribution(self.N_DN + self.M_CN)
        distributed = distributed * self.TOTAL_BW  # 无人机的总带宽
        CN_len = self.N_DN - 1
        all_L_c = rng.choice([3, 6, 9], size=self.N_DN, p=[0.4, 0.5, 0.1])
        # all_L_c = [9,3,6,3,3]
        # all_L_c = [6, 6, 6, 3, 3] #cn=5
        print("DEBUG : all_L_c is ", all_L_c)
        for i in range(self.N_DN):
            # p 的总和必须为 1，我们假设随机选择分割层
            L_c = all_L_c[i]
            dn = DataNode(self.cfg, L_c)
            dn.DN2CN_SINR = self.base_channel_quality
            dn.CN2DN_SINR = self.base_channel_quality
            dn.DN2UAV_SINR = self.DN2UAVSINR[i]
            dn.UAV2DN_SINR = self.UAV2DNSINR
            dn.DN2CN_W = self.cfg.DN2CN_W
            dn.CN2DN_W = self.cfg.CN2DN_W
            dn.CN2UAV_W = distributed[i]
            dn.UAV2DN_W = self.cfg.TOTAL_BW
            dn.compute_unit_cost()  # 计算一下单位能耗
            self.DN_list.append(dn)

        # 1. 先根据 cost 升序排序
        self.DN_list.sort(key=lambda x: x.unit_cost)
        unit_cost = []
        unit_time = []
        # 2. 遍历排序后的列表，重新赋予 ID
        for new_id, dn in enumerate(self.DN_list):
            dn.id = new_id
            unit_cost.append(dn.unit_cost)
            unit_time.append(dn.unit_time)

        print("unit_cost:", unit_cost)
        print("unit_time:", unit_time)

        # 初始化算力节点
        # 获取可行的算力余量以及对应的类型
        # energy_list,type_list = self.energy_and_type()
        # 现在默认取值
        energy_list = [0] * self.M_CN  # 暂时没用到，先不管他
        type_list = [1.0 - (i + 1) * 0.05 for i in range(self.M_CN)]  # 随机分配类型
        # freq_list = [self.base_freq-(i+1)*1.25e9 for i in range(self.M_CN)]

        for i in range(self.M_CN):
            cn = ComputeNode(self.cfg, self.max_freq, energy_list[i], type_list[i], i + 1)
            cn.CN2DN_SINR = self.base_channel_quality
            cn.CN2UAV_SINR = self.CN2UAVSINR[i]
            cn.CN2UAV_W = distributed[CN_len + i]
            # 这两个直接赋值
            cn.DN2CN_W = self.cfg.DN2CN_W
            cn.CN2DN_W = self.cfg.CN2DN_W
            cn.UAV2CN_W = self.cfg.TOTAL_BW
            self.CN_list.append(cn)

    def energy_and_type(self):
        # 设定 5 个具体的算力节点 (焦耳)
        nodes_theta = np.array([5500, 7000, 10000, 14000, 18000])
        node_names = ['N1', 'N2', 'N3', 'N4', 'N5']

        # 实时计算标准差 (Sigma)
        current_sigma = np.std(nodes_theta)
        k = 1.0
        for i in nodes_theta:
            typey = self.func_preference(nodes_theta[i], k, current_sigma)
            print(typey)
        energy_list = []
        type_list = []
        return energy_list, type_list

    """
    类型计算公式
    """

    def func_preference(self, theta, k, sigma):
        return 1 / (1 + np.exp(-k * (theta / sigma)))

    def init_UAV(self):
        self.uav = UAV(self.cfg)
        self.uav.DN_list = self.DN_list
        self.uav.CN_list = self.CN_list

    def init_all_SINR(self, flag=False):
        if flag:
            np.random.seed(42)
        # 无人机的信噪比
        self.UAV2DNSINR = np.random.normal(1.4, 0.4, size=1)  # 生成一个符合均值为1，标准差为0.4的随机数
        self.UAV2DNSINR = np.clip(self.UAV2DNSINR, 0.8, 2)  #

        self.DN2UAVSINR = np.random.normal(1.4, 0.4, size=self.N_DN)  # 生成verifier_nums-1个，均值为1.4，标准差为0.4的随机数
        self.DN2UAVSINR = np.clip(self.DN2UAVSINR, 0.8, 2)

        self.CN2UAVSINR = np.random.normal(1.4, 0.4, size=self.M_CN)
        self.CN2UAVSINR = np.clip(self.CN2UAVSINR, 0.8, 2)

        self.DN2CNSINR = np.random.normal(1.4, 0.4, size=self.N_DN)
        self.DN2CNSINR = np.clip(self.DN2CNSINR, 0.8, 2)

        self.CN2DNSINR = np.random.normal(1.4, 0.4, size=self.M_CN)
        self.CN2DNSINR = np.clip(self.CN2DNSINR, 0.8, 2)

        # 这里暂时设置为相同的信噪比
        base_channel_quality = np.random.normal(1.4, 0.4, size=1)
        self.base_channel_quality = np.clip(base_channel_quality, 0.8, 2.0)

    def Modification_SINR(self):
        for i, dn in enumerate(self.DN_list):
            dn.DN2CN_SINR = self.DN2CNSINR[i]
            dn.DN2UAV_SINR = self.DN2UAVSINR[i]
            dn.UAV2DN_SINR = self.UAV2DNSINR

        for i, cn in enumerate(self.CN_list):
            cn.CN2DN_SINR = self.CN2DNSINR[i]
            cn.CN2UAV_SINR = self.CN2UAVSINR[i]

    def reset(self):
        # 重置信噪比
        self.init_all_SINR()
        self.Modification_SINR()
        # 初始化时随机设置ir和ic状态
        self.dn_ir = np.random.choice([0, 1], size=self.N_DN)
        # self.dn_ic = np.random.choice([0, 1], size=self.N_DN)

        self.cn_ir = np.random.choice([0, 1], size=self.M_CN)
        # self.cn_ic = np.random.choice([0, 1], size=self.M_CN)

        self.state = np.hstack([self.dn_ir, self.cn_ir, self.DN2UAVSINR, self.CN2UAVSINR])
        # 重置节点和对象状态
        for i, dn in enumerate(self.DN_list):
            dn.reset()

        for i, cn in enumerate(self.CN_list):
            cn.reset()

        self.uav.reset()

        # 计算状态
        return self.state

    def node_reset(self):
        # 重置节点和对象状态
        for i, dn in enumerate(self.DN_list):
            dn.reset()

        for i, cn in enumerate(self.CN_list):
            cn.reset()

        self.uav.reset()

    def decode_action(self, proc_cont):
        """
        动作解析与合同生成 (Deep Logic)
        输入:
            proc_cont: [Dn(N), fm(M), W(N+M)] (归一化值)
            disc_action: [Routing(N)]
        输出:
            包含推导出的 Rn 和 Rm 的完整物理动作
        """
        N = self.cfg.N_DN
        M = self.cfg.M_CN

        # 1. 提取基础物理量
        dn_ascending = proc_cont[0:N]
        Dn_phys = dn_ascending[::-1] * self.MAX_DATA_COUNT
        Dn_phys = np.round(Dn_phys, 0)
        fm_phys = proc_cont[N:N + M]
        fm_phys = fm_phys * 0.9 + 0.1
        bw_ratios = proc_cont[N + M:-M]
        W_phys = bw_ratios * self.TOTAL_BW
        W = W_phys[:N]
        Rn_desc = self.calculate_dn_contract(Dn_phys, N, W)

        # 映射回原始顺序
        for k in range(N):
            self.DN_list[k].D_req = Dn_phys[k]

        ### 获取CN的合同
        # 4.解析算力权重 (用于贪婪路由)
        cn_weights = proc_cont[-M:]  # 最后 M 个是权重
        # 获取对应分组以及分组的dn
        group, cn_to_dn_list = self.weighted_greedy_routing(self.DN_list, cn_weights, M)
        H_alpha_list, group_idx = self.sort_Cn_H(cn_to_dn_list)
        # 2. 对负载进行排序 (负载重者，视为分配给了高能耗余量的节点)
        # 高类型节点应该获取大的负载任务，所以这里应该设置为降序
        sort_idx = np.argsort(H_alpha_list)[::-1]
        group_idx_arr = np.array(group_idx, dtype=object)
        sort_group_idx = group_idx_arr[sort_idx]
        fm_sort = fm_phys[np.argsort(fm_phys)[::-1]]
        # 获取分组并排序之后的dn_list
        sort_dn_list = [[] for _ in range(M)]
        # 遍历每个数据节点，将其"扔"进对应的算力节点桶里
        for cn_idx, dn_idx in enumerate(sort_group_idx):
            for id in dn_idx:
                # 记录归属关系 (打包)
                sort_dn_list[cn_idx].append(self.DN_list[id])  # 将对应的DN存给对应的cn

        self.sort_idx = sort_idx
        W = W_phys[N:]
        Rm = self.calculate_cn_contract(M, W, fm_sort, sort_dn_list)

        return {
            'Dn': Dn_phys, 'Rn': Rn_desc,
            'Rm': Rm, 'fm': fm_sort,
            'cn2dn_list': sort_dn_list,
            'bandwidth': W_phys,
            'group': sort_group_idx
        }

    def calculate_cn_contract(self, M, W_phys, fm_sort, sort_dn_list):
        Rm = np.zeros(M)
        Uti_CN = np.zeros(M)
        # 计算对应的能耗
        cost_list = self.Cn_cost(sort_dn_list, fm_sort, W_phys)
        cost_self = np.zeros(M)
        for k in range(M - 1, -1, -1):
            # 获取最后对应位置的数据节点
            cn = self.CN_list[k]
            w = W_phys[k]
            cn.CN2UAV_W = w
            # 计算合理的数据量对应的能耗
            cost_curr_own = cost_list[k]
            cost_self[k] = cost_curr_own
            # 先生成满足IC的合同
            if k == M - 1:
                # --- Base Case: 最差类型 (Type Worst) ---
                # 约束：IR (个人理性) 紧致 -> 效用为 0
                # R = Cost * Data
                Uti_CN[k] = 0
                Rm[k] = cost_curr_own

            else:
                # 获取下一级的合同的类型
                dn_list_ic = sort_dn_list[k + 1]
                fm_ic = fm_sort[k + 1]
                if fm_ic <= 1:
                    fm_ic *= cn.max_freq
                # 上一个节点的能耗
                cost_worse = cost_self[k + 1]
                # 当前节点获取上一个节点的合同的能耗
                cost_curr_mimic, _ = cn.evaluate_contract(w, dn_list_ic, fm_ic)
                # 计算租金增量：(上个节点成本 - 当前节点选择成本) * 任务量
                rent_increment = cost_worse - cost_curr_mimic
                if rent_increment < 0:
                    test = rent_increment
                    # 假设上一个排序为[1,2,0]那么我们是无法生成满足对应IC的合同？
                    # 因为0这个节点的type本身比2要大，那么对应的cost*（1/type）就必然会比节点2要小
                    # 那么就会导致同样的合同，给0的cost比给2要小，导致rent_increment为负数？
                    # cost_curr_mimic, _ = cn.evaluate_contract(w, dn_list_ic, fm_ic)
                    print("rent_increment :", test)

                # 当前好人的总效用 = 差人的效用 + 新增租金
                Uti_CN[k] = Uti_CN[k + 1] + rent_increment
                # 最终激励 = 物理成本 + 净效用(租金)
                Rm[k] = cost_curr_own + Uti_CN[k]
        return Rm

    def calculate_dn_contract(self, Dn_phys, N, W_phys):
        Rn_desc = np.zeros(N)
        Uti_DN = np.zeros(N)  # 记录净效用
        # 如果需要的话，应该如何分配？优先给算力更大的带宽？
        # 从最差的节点 (Index N-1) 开始，倒着推到最好的节点 (Index 0)
        for k in range(N - 1, -1, -1):
            # 获取最后对应位置的数据节点
            dn = self.DN_list[k]
            w = W_phys[k]
            dn.DN2UAV_W = w
            # 当前合同的理论数据量
            data_curr = Dn_phys[k]
            # 计算合理的数据量对应的能耗
            cost_curr_own = dn.evaluate_contract(w, data_curr, Flag=True)

            # 先生成满足IC的合同
            if k == N - 1:
                # --- Base Case: 最差类型 (Type Worst) ---
                # 约束：IR (个人理性) 紧致 -> 效用为 0
                # R = Cost * Data
                Uti_DN[k] = 0
                Rn_desc[k] = cost_curr_own

            else:
                # 获取下一级的合同的类型
                dn_ic = self.DN_list[k + 1]
                dn_copy = copy.deepcopy(dn_ic)
                dn_copy.unit_cost = dn.unit_cost

                w_ic = W_phys[k + 1]
                data_ic = Dn_phys[k + 1]

                cost_worse = dn_ic.evaluate_contract(w_ic, data_ic)

                # 当前节点获取下一个节点的合同的能耗
                cost_curr_mimic = dn_copy.evaluate_contract(w_ic, data_ic)

                # 计算租金增量：(差人成本 - 好人成本) * 差人任务量
                rent_increment = cost_worse - cost_curr_mimic
                if rent_increment < 0:
                    print("DN is error!!!")

                # 当前好人的总效用 = 差人的效用 + 新增租金
                Uti_DN[k] = Uti_DN[k + 1] + rent_increment

                # 最终激励 = 物理成本 + 净效用(租金)
                Rn_desc[k] = cost_curr_own + Uti_DN[k]
        return Rn_desc

    def calculate_utility_matrix(self, action):
        '''
        计算效用矩阵，并同时验证 IC (激励相容) 约束的满足概率。

        返回:
            dn_utility_matrix (ndarray): 数据节点效用矩阵 N x N
            cn_utility_matrix (ndarray): 算力节点效用矩阵 M x M
            dn_ic_rate (float): 数据节点满足 IC 的比例 [0.0 ~ 1.0]
            cn_ic_rate (float): 算力节点满足 IC 的比例 [0.0 ~ 1.0]
        '''
        N = self.N_DN
        M = self.M_CN

        # 1. 提取动作参数
        Dn = action['Dn']
        Rn = action['Rn']
        Rm = action['Rm']
        Fm = action['fm']
        cn2dn_list = action['cn2dn_list']
        dn_bw = action['bandwidth'][:N]
        cn_bw = action['bandwidth'][N:]

        # 2. 初始化矩阵
        dn_utility_matrix = np.zeros((N, N))
        cn_utility_matrix = np.zeros((M, M))

        # 用于记录满足 IC 的节点数量
        dn_ic_count = 0
        cn_ic_count = 0

        dn_ic = np.ones(N)
        cn_ic = np.ones(M)
        # 容忍浮点数误差的阈值 (非常关键，否则可能因为 0.0000001 的差别被误判为违反 IC)
        TOLERANCE = 1e-5

        # =========================================================
        # 3. 评估数据节点 (DN)
        # =========================================================
        for i in range(N):
            dn_node = self.DN_list[i]  # 物理节点 i

            # 计算该节点面对所有 N 个合同的效用
            for j in range(N):
                bw = dn_bw[j]
                d_req = Dn[j]
                r_offer = Rn[j]
                # 注意参数顺序要与 evaluate_contract 定义一致
                uti = dn_node.evaluate_contract(bw, d_req, r_offer, Flag=False)
                dn_utility_matrix[i, j] = uti

            # --- 验证 DN 的 IC ---
            # 获取节点 i 选自己合同的效用 (对角线元素)
            u_own = dn_utility_matrix[i, i]

            # 检查是否有任何其他合同的效用 严格大于 自己的合同
            # 如果没有，说明满足 IC
            is_ic_satisfied = True
            for j in range(N):
                if i != j and dn_utility_matrix[i, j] > u_own + TOLERANCE:
                    is_ic_satisfied = False
                    dn_ic[i] = 0
                    break  # 只要发现一个更好的，就违反了 IC

            if is_ic_satisfied:
                dn_ic_count += 1

        # =========================================================
        # 4. 评估算力节点 (CN)
        # =========================================================
        for i in range(M):
            cn_node = self.CN_list[i]
            bw_i = cn_bw[i]  # 注意：在评估其他合同时，带宽是用自己的还是合同里的？通常是用合同配好的带宽。
            # 这里我按您的代码逻辑，使用对应合同 j 的带宽会更合理，或者使用全局平均带宽。
            # 假设这里使用该节点自身被分配的带宽 (与您的原代码保持一致)

            for j in range(M):
                dn_list = cn2dn_list[j]
                rm = Rm[j]
                fm = Fm[j]

                uti = cn_node.evaluate_contract(bw_i, dn_list, fm, flag=False, R_offer=rm)
                cn_utility_matrix[i, j] = uti

            # --- 验证 CN 的 IC ---
            u_own = cn_utility_matrix[i, i]
            is_ic_satisfied = True

            for j in range(M):
                if i != j and cn_utility_matrix[i, j] > u_own + TOLERANCE:
                    is_ic_satisfied = False
                    cn_ic[i] = 0
                    break

            if is_ic_satisfied:
                cn_ic_count += 1

        # =========================================================
        # 5. 计算概率并返回
        # =========================================================
        dn_ic_rate = dn_ic_count / N
        cn_ic_rate = cn_ic_count / M

        return dn_utility_matrix, cn_utility_matrix, dn_ic_rate, cn_ic_rate,dn_ic,cn_ic

    def weighted_greedy_routing(self, Dn_list, node_weights, num_cn):
        """
        输入:
            Dn_list: 数据节点的数据量列表 [D0, D1, D2, D3, D4]
            node_weights: PPO输出的算力节点权重 [w0, w1, w2] (值越大代表越推荐)
            num_cn: 3
        输出:
            routing: [0, 2, 0, 1, 2]
        """
        # 1. 预处理权重 (映射到 0.1 ~ 2.0，避免除零)
        # PPO输出是0~1，我们把它拉伸一下，让调节范围更大
        real_weights = node_weights * 2.0 + 0.1

        # 这里记录的是已经分配的任务总量
        cn_loads = np.zeros(num_cn)

        # 结果容器
        group = np.zeros(len(Dn_list), dtype=int)

        # 4. 贪婪循环
        for dn_idx, dn in enumerate(Dn_list):
            H_alpha = dn.body_flops * dn.D_req  # 将MFLOPS转化为flops
            # --- 核心决策公式 ---
            # 我们要找通过加权后，"感知成本"增加最小的那个节点
            # Cost = (CurrentLoad + NewData) / Weight
            # Weight 越大，Cost 越小，越容易被选中

            costs = (cn_loads + H_alpha) / real_weights

            # 选 Cost 最小的
            best_cn = np.argmin(costs)

            # 执行分配
            group[dn_idx] = best_cn
            cn_loads[best_cn] += H_alpha

        # 初始化
        cn_to_dn_list = [[] for _ in range(num_cn)]

        # 遍历每个数据节点，将其"扔"进对应的算力节点桶里
        for dn_idx, cn_idx in enumerate(group):
            # 1. 记录归属关系 (打包)
            cn_to_dn_list[cn_idx].append(Dn_list[dn_idx])  # 将对应的DN存给对应的cn

        return group, cn_to_dn_list

    def Cn_cost(self, cn_to_dn_list, cn_fm, W_phys):
        cost_list = []
        # 这里计算的是相对大小不是精准的能耗
        for i in range(self.M_CN):
            dn_list = cn_to_dn_list[i]
            if cn_fm[i] <= 1:
                fm = cn_fm[i] * self.CN_list[i].max_freq
            else:
                fm = cn_fm[i]
            w = W_phys[i]
            total_E, H_alpha = self.CN_list[i].evaluate_contract(w, dn_list, fm, flag=True)
            cost_list.append(total_E.item())
        return cost_list

    def sort_Cn_H(self, cn_to_dn_list):
        H_alpha_list = []
        group_idx = []
        # 这里计算的是相对大小不是精准的能耗
        for i in range(self.M_CN):
            dn_list = cn_to_dn_list[i]
            dn_list_id = []
            H_alpha = 0.0
            for dn in dn_list:
                dn_list_id.append(dn.id)
                H_alpha += dn.body_flops * dn.D_req  # 这里是MFlops
            H_alpha_list.append(H_alpha / 1e3)  # 将MFlops转化为GFlops
            group_idx.append(dn_list_id)
        return H_alpha_list, group_idx

    def step(self, proc_cont):
        # 将action分解出来，并分别计算对应的奖励和动作

        # 这里对应的合同已经满足IC了，但是分配有问题，因此需要检测IR
        all_action = self.decode_action(proc_cont)

        # 计算效用和状态
        dn_contract, cn_contract = self.action_to_contract(all_action)
        dn_utility_matrix, cn_utility_matrix, dn_ic_rate, cn_ic_rate,dn_ic,cn_ic = self.calculate_utility_matrix(all_action)
        compliance_rate, violations, ic_list = self.check_ic_status(all_action)
        print('compliance_rate:', compliance_rate)
        self.cn_ic = cn_ic
        self.dn_ic = dn_ic
        uti_, dn_ir, cn_ir, uav_dn_list, uav_cn_list = self.calculate_Utility(all_action, dn_ic,cn_ic)
        uav_info = self.uav.calculate_cost(uav_dn_list, uav_cn_list)
        # print(self.uav.total_cost,self.uav.total_time)
        self.uav.utility = uti_['uav_uti'] - uav_info['total_cost']
        self.state = np.hstack([self.dn_ir, self.cn_ir, self.DN2UAVSINR, self.CN2UAVSINR])
        self.reward = self.uav.utility
        if self.reward < 0:
            uav_info = self.uav.calculate_cost(uav_dn_list, uav_cn_list)
            print(f"self.reward is {self.reward}")
        total_data = sum(all_action['Dn'])
        return self.state, float(self.reward), False, uav_info, dn_contract, cn_contract, uti_, total_data,all_action

    def check_ic_status(self, action_dict):
        """
        检测当前分配方案是否满足全局 IC 约束
        返回:
          - ic_compliance_rate: 满足 IC 的节点比例
          - violations: 违约详情列表
        """
        # 1. 提取所有生成的合同菜单
        # 这里需要把 Dn, Rn 和 Rm, Beta, fm 组合成合同列表
        # 假设我们只检查算力节点 (CN) 的 IC

        # 算力合同菜单列表: [(Load_0, R_0), (Load_1, R_1), ...]
        # Load = 物理负载 (beta, f)
        beta_list = action_dict['cn2dn_list']
        fm_list = action_dict['fm']
        rm_list = action_dict['Rm']  # 总激励

        M = self.M_CN
        ic_count = 0
        violations = []
        ic_list = np.ones(M)
        own_uti = []

        # 遍历每个物理节点
        for m in range(M):
            cn_node = self.CN_list[m]
            w = cn_node.CN2UAV_W
            u_own = rm_list[m] - (1 / cn_node.type) * cn_node.total_E
            own_uti.append(u_own)
            # B. 遍历菜单里"别人的合同"，看有没有更高激励的
            is_ic_satisfied = True

            for j in range(M):
                if m == j: continue

                other_beta = beta_list[j]
                other_fm = fm_list[j]
                other_pay = rm_list[j]

                # 计算如果我选 j 的效用
                # CN2UAV_W,DN_list,f_m,flag=False,R_offer=None
                u_mimic = cn_node.evaluate_contract(w, other_beta, other_fm, flag=False, R_offer=other_pay)

                if u_mimic > u_own + 1e-5:  # 加微小阈值防浮点误差
                    is_ic_satisfied = False
                    ic_list[m] = 0
                    violations.append(f"Node {m} envies Node {j} (Gain: {float(u_mimic - u_own):.4f})")
                    u_mimic = cn_node.evaluate_contract(w, other_beta, other_fm, flag=False, R_offer=other_pay)
                    break  # 只要羡慕一个，就算 IC 失败

            if is_ic_satisfied:
                ic_count += 1

        compliance_rate = ic_count / M
        return compliance_rate, violations, ic_list

    def action_to_contract(self, all_action):
        # 根据动作去解析一下合同，并打印出来，判断是否合理？
        dn_contract = list(list(all_action['Dn']) + list(np.round(all_action['Rn'], 2)))
        fm = []
        H_alpha = []
        # print(self.sort_idx)
        for i, cn in enumerate(self.CN_list):
            fm.append(np.round((cn.max_freq * all_action['fm'][i] / 1e9), 2))
            H_alpha.append(np.round(cn.H_alpha, 2))
        # print(all_action['routing'])
        # 改为这行 (将其递归地转化为原生 Python List)：
        group_idx = [x.tolist() if isinstance(x, np.ndarray) else list(x) for x in all_action['group']]
        cn_contract = group_idx + list(H_alpha) + list(fm) + list(all_action['Rm'])
        return dn_contract, cn_contract

    def calculate_Utility(self, all_action, dn_ic,cn_ic, flag=False):
        uav_utility = 0
        Dn = all_action['Dn']
        Rn = all_action['Rn']
        W = all_action['bandwidth']
        dn_ir = np.zeros(self.N_DN)
        cn_ir = np.zeros(self.M_CN)
        uav_dn_list = []
        dn_uti = 0.0
        cn_uti = 0.0
        for i, dn in enumerate(self.DN_list):
            dn.D_req = Dn[i]
            dn.R_offer = Rn[i]
            w = W[i]
            uti = dn.evaluate_contract(w, dn.D_req, dn.R_offer, flag)
            if uti < 0:
                dn_uti += uti
                dn_ir[i] = 0
            elif dn_ic[i] == 0:
                dn_uti -= 100
            else:
                dn_uti += self.tau_dn * self.uav.DN_uti(dn)
                dn_ir[i] = 1
                # 如果符合ir则把他添加进来
                uav_dn_list.append(dn)

        dn_list = all_action['cn2dn_list']
        fm = all_action['fm']
        Rm = all_action['Rm']
        uav_cn_list = []
        for i, cn in enumerate(self.CN_list):
            cn.DN_list = dn_list[i]
            cn.fm = fm[i]
            w = W[self.N_DN + i]
            uti = cn.evaluate_contract(w, cn.DN_list, cn.fm, flag=False, R_offer=Rm[i])
            if uti < 0:
                cn_uti += uti
                cn_ir[i] = 0
            elif cn_ic[i] == 0:
                cn_uti -= 100
            else:
                cn_uti += self.tau_cn * self.uav.CN_uti(cn)
                cn_ir[i] = 1
                uav_cn_list.append(cn)

        if dn_uti < 0 or cn_uti < 0:
            uav_utility = dn_uti if dn_uti < 0 else cn_uti
        else:
            uav_utility = dn_uti + cn_uti
        uti_all = {
            'uav_uti': uav_utility,
            'dn_uti': dn_uti,
            'cn_uti': cn_uti
        }
        return uti_all, dn_ir, cn_ir, uav_dn_list, uav_cn_list

    def step2(self, contract):
        # 普通合同走这个通道来计算效用
        # 合同输入形式
        """
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
        """
        # 1.根据合同计算时间
        N = self.N_DN
        M = self.M_CN
        W_n = contract['bandwidth'][:N]
        W_m = contract['bandwidth'][N:]
        cn2dn_list = self.get_dn_list(contract['routing'])
        contract['cn2dn_list'] = cn2dn_list
        contract['Rn'] = self.calculate_dn_contract(contract['Dn'], N, W_n)
        for i, dn in enumerate(self.DN_list):
            dn.D_req = contract['Dn'][i]
        contract['Rm'] = self.calculate_cn_contract(M, W_m, contract['fm'], cn2dn_list)

        dn_contract = list(list(contract['Dn']) + list(np.round(contract['Rn'], 2)))
        fm = np.round(contract['fm'] / 1e9, 2)
        H_alpha = np.round(contract['beta_m'] / 1e3, 2)
        cn_contract = list(contract['routing']) + list(H_alpha) + list(fm) + list(contract['Rm'])
        dn_utility_matrix, cn_utility_matrix, dn_ic_rate, cn_ic_rate,dn_ic,cn_ic = self.calculate_utility_matrix(contract)

        uti_, dn_ir, cn_ir, uav_dn_list, uav_cn_list = self.calculate_Utility(contract, dn_ic,cn_ic)

        uav_info = self.uav.calculate_cost(uav_dn_list, uav_cn_list)

        self.uav.utility = uti_['uav_uti'] - uav_info['total_cost']

        self.state = np.hstack([self.dn_ir, self.cn_ir, self.DN2UAVSINR, self.CN2UAVSINR])
        self.reward = self.uav.utility
        total_data = sum(contract['Dn'])
        return self.state, float(self.reward), False, uav_info, dn_contract, cn_contract, uti_, total_data

    def get_dn_list(self, routing):
        cn2dn_list = []
        for route_idx, route in enumerate(routing):
            # ========== 核心兼容逻辑 ==========
            # 步骤1：将numpy数值转为Python原生int，避免类型问题
            if isinstance(route, np.integer):  # 匹配numpy.int32/int64等类型
                route = int(route)
            # 步骤2：统一转为可迭代的列表
            if isinstance(route, (int, float)):  # 如果是单个数值
                route_iter = [route]  # 转为单元素列表（可迭代）
            else:  # 如果本身就是可迭代类型（列表/元组等）
                route_iter = route

            # ========== 原有业务逻辑 ==========
            for _, dn in enumerate(self.DN_list):
                if route_idx in route_iter:
                    cn2dn_list.append([dn])

        return cn2dn_list

    def step3(self, action):
        '''
        return {
        'Dn': Dn_phys, 'Rn': Rn_phys,
        'Rm': Rm_total, 'fm': fm_phys,
        'bandwidth': W_alloc, 'routing': routing,
        'beta_m': beta_m, 'mode': 'UNIFORM_PRICING'
        }
        '''
        dn_pricing = list(list(action['Dn']) + list(np.round(action['Rn'], 2)))
        fm = np.round(action['fm'] / 1e9, 2)
        H_alpha = np.round(action['beta_m'], 2)
        cn_pricing = list(action['routing']) + list(H_alpha) + list(fm) + list(action['Rm'])
        cn2dn_list = self.get_dn_list(action['routing'])
        action['cn2dn_list'] = cn2dn_list
        dn_ir, cn_ir, uti_ = self.calculate_pricing_utility(action)
        uav_info = self.pricing_calculate_cost(dn_ir, cn_ir)

        self.uav.utility = uti_['uav_uti'] - uav_info['total_cost']

        self.state = np.hstack([self.dn_ir, self.cn_ir, self.DN2UAVSINR, self.CN2UAVSINR])
        self.reward = self.uav.utility

        total_data = 0
        for i, ir in enumerate(dn_ir):
            if ir == 1:
                total_data += action['Dn'][i]
        return self.state, float(self.reward), False, uav_info, dn_pricing, cn_pricing, uti_, total_data

    def calculate_pricing_utility(self, action):
        Dn = action['Dn']
        Rn = action['Rn']
        dn_ir = np.zeros(self.N_DN)
        dn_w = action['bandwidth'][:self.N_DN]
        dn_uti = 0.0
        for i, dn in enumerate(self.DN_list):
            dn.D_req = Dn[i]
            dn.R_offer = Rn[i]
            uti = dn.evaluate_contract(dn_w[i], Dn[i], Rn[i], True)
            if uti < 0:
                dn_uti += 0
                dn_ir[i] = 0
            else:
                dn_uti += self.tau_dn * self.uav.DN_uti(dn)
                dn_ir[i] = 1

        fm = action['fm']
        cn2dn_list = action['cn2dn_list']
        Rm = action['Rm']
        cn_w = action['bandwidth'][self.N_DN:]
        cn_ir = np.zeros(self.M_CN)
        cn_uti = 0.0
        for j, cn in enumerate(self.CN_list):
            dn_list = cn2dn_list[j]
            cn.fm = fm[j]
            cn.R_offer = Rm[j]
            #  CN2UAV_W,DN_list,f_m,flag=False,R_offer=None
            uti = cn.evaluate_contract(cn_w[j], dn_list, fm[j], True, Rm[j])
            if uti < 0:
                cn_uti += 0
                cn_ir[j] = 0
            else:
                cn_uti += self.tau_cn * self.uav.CN_uti(cn)
                cn_ir[j] = 1
        uav_utility = dn_uti + cn_uti
        uti_all = {
            'uav_uti': uav_utility,
            'dn_uti': dn_uti,
            'cn_uti': cn_uti
        }
        return dn_ir, cn_ir, uti_all

    def pricing_calculate_cost(self, dn_ir, cn_ir):
        self.uav.total_cost = 0.0
        self.uav.total_time = 0.0
        self.uav.total_com_energy = 0.0
        self.uav.total_time = 0.0

        DN_list = self.DN_list
        CN_list = self.CN_list
        total_dn_time = 0.0
        for i, dn in enumerate(DN_list):
            if dn_ir[i] == 1:
                self.uav.total_com_energy += self.uav.P_to_DN * dn.UAV2DN_Com_Time + self.uav.P_from_DN * dn.DN2UAV_Com_Time
                total_dn_time = max(dn.T * dn.cmp_time * dn.D_req + dn.other_time, total_dn_time)

        total_cn_time = 0.0
        for j,cn in enumerate(CN_list):
            if cn_ir[j] == 1:
                self.uav.total_com_energy += self.uav.P_from_CN * cn.CN2UAV_com_time
                total_cn_time = max(cn.total_time, total_cn_time)

        self.uav.total_time = total_dn_time + total_cn_time
        agg_cost = self.uav.mu * (self.uav.f_p * self.uav.kappa) ** 2 * 1e6
        self.uav.total_cost = self.uav.total_com_energy + agg_cost + self.uav.E_h * self.uav.total_time
        # if self.total_cost > self.max_total_cost:
        #     self.max_total_cost = self.total_cost
        #     print(f"the max total cost is {self.max_total_cost}")
        uav_info = {
            'total_time': self.uav.total_time,
            'total_cost': self.uav.total_cost,
            'dn_time': total_dn_time,
            'cn_time': total_cn_time
        }
        return uav_info


def energy_and_type():
    # 设定 5 个具体的算力节点 (焦耳)
    nodes_theta = np.array([37000, 50000, 55000, 46000, 60000])
    node_names = ['N1', 'N2', 'N3', 'N4', 'N5']

    # 实时计算标准差 (Sigma)
    current_sigma = np.std(nodes_theta)
    k = 0.5
    d = 1.0
    for i, node in enumerate(nodes_theta):
        type = np.round(func_preference(node, k, current_sigma), 3)
        print("type is ", type, "1/ type is ", 2 / type)
        print("d", 1 / (d - (i + 1) * 0.08))

    energy_list = []
    type_list = []
    return energy_list, type_list


"""
类型计算公式
"""


def func_preference(theta, k, sigma):
    return 1 / (1 + np.exp(-k * (theta / sigma)))


if __name__ == '__main__':
    energy_and_type()
