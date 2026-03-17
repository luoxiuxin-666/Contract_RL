
from UsualFunctions import LOG,CommonFun
import numpy as np
# ==============================================================================
# 配置参数
# ==============================================================================
class Config:
    def __init__(self):
        # --- 系统规模 ---
        self.N_DN = 5  # 数据节点数
        self.M_CN = 5  # 算力节点数
        self.SFL_CONTRACT = True
        self.SFL_PRICING = True
        self.FL_CONTRACT = True
        self.FL_PRICING = True
        self.mode = 'SFL' # mode = SFL or FL
        self.UNIT_FULL_FLOPS = 1089
        self.MODEL_SIZE_MBITS = 288

        # --- 物理约束 (映射常数) ---
        self.max_data_size = 120 #MB 数据的总大小为120MB
        self.max_data_count = 10000
        self.MB_unit_ = 8e6 #1 MB=8e^6 bits
        self.f_n = np.array([1, 1.25]) * 1e9  # GHz (fm) 1GHz = 1e9Hz
        self.kappa = 16 #1bts数据所需要的运行次数
        self.T = 1
        self.mu_dn = 1e-28 #硬件系数
        self.TOTAL_BW = 2*1e8  # 200MHz (Bandwidth),2*10e8
        self.max_freq = 10e9
        self.DN2CN_W = 1.5e7 #1MHz = 10^6Hz
        self.CN2DN_W = 1.5e7 #2MHz

        #数据节点的通信功率:单位W（watt）
        self.DN_P_1 = 0.2
        self.DN_P_2 = 0.4
        self.P_DN2CN = self.DN_P_1
        self.P_DNFCN = self.DN_P_1
        self.P_DN2UAV = self.DN_P_1
        self.P_DNFUAV = self.DN_P_1

        # --- 算力节点数据---
        self.cn_total_energy = 111
        self.mu_m = 1e-28
        self.CN_P = 0.2
        # 默认都是0.2W
        self.P_CN2DN = self.CN_P
        self.P_CNFDN = self.CN_P
        self.P_CN2UAV = self.CN_P
        self.P_CNFUAV = self.CN_P
        # CPU设置为5-10GHz

        # --- 无人机 ---
        self.beta_1 = 600
        self.beta_2 = 1
        self.alpha_dn = 300
        self.alpha_fl_ppo = 500
        self.alpha_fl_contract = 300
        self.f_p = 1.5*1e9 #无人机的CPU
        self.E_h = 0.5
        self.mu_uav = 1e-28
        #Watt
        self.P_UAV2DN = 1
        self.P_UAVFDN = 1
        self.P_UAVFCN = 1
        # 无人机的下行带宽
        self.UAV_W = 2*1e8  #200MHz

        # --- 环境参数 ---
        self.tau_dn = 1
        self.tau_cn = 1


        # --- 训练参数 ---
        # self.TOTAL_STEPS = self.configDict['total_steps']
        self.TOTAL_EPISODES = 3000
        self.MAX_STEPS_PER_EP = 64
        self.UPDATE_TIMESTEP = 1 # 多少步更新一次？

        # --- PPO 参数 ---
        self.LR_ACTOR = 1e-4
        self.LR_CRITIC = 1e-5
        self.GAMMA = 0.95
        self.K_EPOCHS = 10 #每轮更新多少次
        self.EPS_CLIP = 1

        # --- 动态学习率调度 ---
        self.USE_LR_SCHEDULER = True

        # 多次波动就降速
        self.LR_PATIENCE = 100

        # 每次遇到瓶颈，学习率减半 (0.5)
        self.LR_FACTOR = 0.6

        # 初始 LR 是 3e-4，下限设为 1e-6，保证后期还能微调
        self.LR_MIN = 1e-6

        # 只要比历史最佳高一点点(0.0001)，就算有进步
        self.LR_THRESHOLD = 1e-4

        # --- 日志与保存 ---
        self.LOG_INTERVAL = 10
        self.SAVE_INTERVAL = 500

        self.cn_list = None
        self.dn_list = None
        # 模型参数
        self.L_t = 12 # 默认尾部分割点为第12层（根据模型确定）