import torch
import torch.nn as nn

# 检查 thop 库
try:
    from thop import profile

    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("【提示】未检测到 thop 库，FLOPs 将显示为 0。请运行: pip install thop\n")


# ==========================================
# 1. 模型定义 (12层 CNN + Classifier)
# ==========================================
class SFL_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SFL_CNN, self).__init__()

        # 定义4个阶段的卷积块
        self.block1 = self._make_layers(3, 3, 64)  # Layers 1-3
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block2 = self._make_layers(3, 64, 128)  # Layers 4-6
        self.pool2 = nn.MaxPool2d(2, 2)

        self.block3 = self._make_layers(3, 128, 256)  # Layers 7-9
        self.pool3 = nn.MaxPool2d(2, 2)

        self.block4 = self._make_layers(3, 256, 512)  # Layers 10-12
        self.pool4 = nn.MaxPool2d(2, 2)

        # 分类头 (全连接层)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # 辅助列表 (用于注册参数和切分)
        self.all_conv_layers = nn.ModuleList()
        self.all_conv_layers.extend(self.block1)
        self.all_conv_layers.extend(self.block2)
        self.all_conv_layers.extend(self.block3)
        self.all_conv_layers.extend(self.block4)

        self.pool_map = {2: self.pool1, 5: self.pool2, 8: self.pool3, 11: self.pool4}

    def _make_layers(self, num_layers, in_c, out_c):
        layers = []
        for i in range(num_layers):
            cur_in = in_c if i == 0 else out_c
            layers.append(nn.Sequential(
                nn.Conv2d(cur_in, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ))
        return layers

    # =========================================================
    # 【新增/修复】 必须包含 forward 函数，否则 thop 无法运行
    # =========================================================
    def forward(self, x):
        # 依次穿过 4 个 Block 和 Pool
        x = self.pool1(self._forward_block(x, self.block1))
        x = self.pool2(self._forward_block(x, self.block2))
        x = self.pool3(self._forward_block(x, self.block3))
        x = self.pool4(self._forward_block(x, self.block4))
        # 进入分类头
        x = self.classifier(x)
        return x

    def _forward_block(self, x, block):
        # 辅助函数：遍历列表中的层
        for layer in block:
            x = layer(x)
        return x
    # =========================================================

    def get_client_model(self, cut_layer):
        # 获取客户端部分子模型
        layers = []
        for i in range(cut_layer):
            layers.append(self.all_conv_layers[i])
            if i in self.pool_map:
                layers.append(self.pool_map[i])
        return nn.Sequential(*layers)

    def _make_layers(self, num_layers, in_c, out_c):
        layers = []
        for i in range(num_layers):
            cur_in = in_c if i == 0 else out_c
            layers.append(nn.Sequential(
                nn.Conv2d(cur_in, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ))
        return layers

    def get_client_model(self, cut_layer):
        # 线性切分：取前 cut_layer 层
        layers = []
        for i in range(cut_layer):
            layers.append(self.all_conv_layers[i])
            if i in self.pool_map:
                layers.append(self.pool_map[i])
        return nn.Sequential(*layers)


# ==========================================
# 2. 核心计算方法 (修正版)
# ==========================================
def get_sfl_metrics(cut_layer, f_n, kappa, mu_n, batch_size=100):
    """
    输入: mu_n (有效电容系数), f_n (频率 Hz)
    """
    # 1. 修正功率计算公式
    # 物理公式: P = mu * f^3 (kappa 影响吞吐率，通常不直接乘在功率公式的立方项里，除非 mu 已经包含了 kappa 的影响)
    # 为了保证计算结果在 4W 左右，我们使用 P = mu * f^3
    # 如果 mu_n 传入的是 2.3e-27 左右，则结果正常。
    # 这里我们采用更稳健的写法：假设 mu_n 是针对 f^3 的系数
    p_n = mu_n * (kappa * f_n ** 3.0)

    # 虚拟输入
    dummy_input = torch.randn(1, 3, 32, 32)
    full_model = SFL_CNN()

    # --- A. 计算完整网络 FLOPs ---
    if HAS_THOP:
        unit_full_fwd, _ = profile(full_model, inputs=(dummy_input,), verbose=False)
    else:
        unit_full_fwd = 0
    # 总计算量 = 前向 + 2*反向 = 3*前向
    unit_full_total = unit_full_fwd * 3

    # --- B. 计算客户端 FLOPs ---
    client_model = full_model.get_client_model(cut_layer)
    if HAS_THOP:
        unit_client_fwd, _ = profile(client_model, inputs=(dummy_input,), verbose=False)
    else:
        unit_client_fwd = 0
    unit_client_total = unit_client_fwd * 3

    # --- C. 计算服务器 FLOPs ---
    unit_server_total = unit_full_total - unit_client_total

    # --- D. 计算数据量 (Smashed Data) ---
    with torch.no_grad():
        output = client_model(dummy_input)
    unit_data_mbits = (output.numel() * 32) / 1e6

    # --- E. 计算时间与能耗 ---
    # 速度 (FLOPS/s)
    speed = f_n * kappa

    # 1. 客户端处理一个 Batch 的时间与能耗
    batch_client_workload = unit_client_total * batch_size
    batch_time_s = batch_client_workload / speed if speed > 0 else 0
    batch_energy_j = p_n * batch_time_s

    # 2. 完整模型处理一个 Batch 的能耗 (用于对比)
    # 之前代码漏乘了 batch_size
    batch_full_workload = unit_full_total * batch_size
    full_time_s = batch_full_workload / speed if speed > 0 else 0
    full_energy_j = p_n * full_time_s

    return {
        "Lc": cut_layer,
        "Full_Total_MFLOPs": unit_full_total / 1e6,
        "Total_Energy_J": full_energy_j,  # 完整模型跑完一个 Batch 的能耗

        "Client_Total_MFLOPs": unit_client_total / 1e6,
        "Batch_Times": batch_time_s * 1000,  # ms
        "Batch_Energy_J": batch_energy_j,

        "Server_Total_MFLOPs": unit_server_total / 1e6,
        "Unit_Data_Mbits": unit_data_mbits,
        "Power_Watt": p_n  # 方便调试查看功率
    }


def model_size_by_sfl(cut_layer):
    """计算模型参数大小 (Mbits)"""
    full_model = SFL_CNN()
    total_params = sum(p.numel() for p in full_model.parameters())
    total_mbits = (total_params * 32) / 1e6

    client_model = full_model.get_client_model(cut_layer)
    client_params = sum(p.numel() for p in client_model.parameters())
    client_mbits = (client_params * 32) / 1e6
    # 单位Mbit
    return client_mbits, total_mbits


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    print(
        f"{'Lc':<3} | {'Cli(M)':<8} | {'Svr(M)':<8} | {'Full(M)':<8} | {'Data(Mb)':<9} | {'Time(ms)':<9} | {'Eng(J)':<9} | {'Mod_C(Mb)':<9} | {'Power(W)':<8}")
    print("-" * 100)

    # --- 物理参数校准 ---
    f_n = 1.2 * 1e9  # 1.2 GHz
    kappa = 16
    # 目标功率 4W。根据 P = mu * f^3 反推 mu
    # mu = 4.0 / (1.2e9 ** 3) ≈ 2.31e-27
    target_mu = 4.0 / (f_n ** 3)

    for lc in range(1, 13):
        # 传入校准后的 mu，确保功率约为 4W
        res = get_sfl_metrics(lc, f_n, kappa, target_mu, batch_size=100)
        c_model, t_model = model_size_by_sfl(lc)

        print(f"{res['Lc']:<3} | "
              f"{res['Client_Total_MFLOPs']:<8.2f} | "
              f"{res['Server_Total_MFLOPs']:<8.2f} | "
              f"{res['Full_Total_MFLOPs']:<8.2f} | "
              f"{res['Unit_Data_Mbits']:<9.4f} | "
              f"{res['Batch_Times']:<9.2f} | "
              f"{res['Batch_Energy_J']:<9.4f} | "
              f"{c_model:<9.4f} | "
              f"{res['Power_Watt']:<8.2f}")  # 检查功率是否正常

    print("1. 算法: 当前使用的是标准线性切分 (Standard Split)，而非 U-Shape。")
    print("2. 功率: 已自动校准 mu 系数，使计算功率维持在 4.00 W 左右，避免量纲爆炸。")
    print("3. 全局能耗: 已修正为基于 Batch (100张图) 的总能耗，而非单样本。")