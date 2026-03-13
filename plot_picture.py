import numpy as np
import os
import matplotlib
import math

# 设置无头模式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置目录
ENV_NAME = "SFL_PPO_Contract"
RESULT_DIR = os.path.join("results", ENV_NAME, "plots")
os.makedirs(RESULT_DIR, exist_ok=True)


def smooth_data(data, window_size=10):
    """
    计算滑动平均，用于平滑曲线
    """
    if len(data) < window_size:
        return data

    # 使用卷积计算滑动平均
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='valid')

    # 填充前面被切掉的数据，保持长度一致以便绘图
    # (简单策略：前面几个点用原始数据填充，或者用逐渐增大的窗口平均)
    padding = data[:window_size - 1]
    return np.concatenate([padding, smoothed])


def plot_learning_curves(metrics_dict, current_episode, window_size=20):
    """
    改进版绘图函数：
    1. 自适应子图布局
    2. 双线绘制 (Raw + Smooth)
    3. 自动 X 轴推断
    4. 自动处理不等长嵌套数据（使用末尾元素补齐）
    """
    if not metrics_dict:
        return

    # 1. 确定指标数量和布局
    num_metrics = len(metrics_dict)
    if num_metrics == 0:
        return

    # 自动计算列数和行数 (最多3列)
    cols = 3 if num_metrics >= 3 else num_metrics
    rows = math.ceil(num_metrics / cols)

    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(f'Training Status (Episode {current_episode})', fontsize=16)

    # 统一处理 axes 为列表，方便遍历 (处理只有1个子图的情况)
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    keys = list(metrics_dict.keys())

    # 2. 遍历绘制每个指标
    for i, key in enumerate(keys):
        ax = axes[i]
        data_list = metrics_dict[key]

        if not data_list:
            continue

        # ==================== 核心修改区域 ====================
        # 步骤 1：寻找当前指标数据中最长的子序列长度
        lengths = [len(item) if isinstance(item, (list, tuple, np.ndarray)) else 1 for item in data_list]
        if not lengths:
            continue
        max_len = max(lengths)

        # 步骤 2：使用最后一个元素对齐补全短数据
        padded_data = []
        for item in data_list:
            if not isinstance(item, (list, tuple, np.ndarray)):
                item_list = [item]  # 将单个数字包装成列表
            else:
                item_list = list(item)

            # 如果遇到空列表，用 NaN 填满；否则用原列表最后一个元素补齐至 max_len
            if len(item_list) == 0:
                item_list = [np.nan] * max_len
            elif len(item_list) < max_len:
                last_element = item_list[-1]
                padding_length = max_len - len(item_list)
                item_list.extend([last_element] * padding_length)

            padded_data.append(item_list)

        # 步骤 3：转换为规范的 float 数组并展平
        raw_data = np.array(padded_data, dtype=np.float64).flatten()
        # ====================================================

        # 检查是否为空
        if len(raw_data) == 0:
            continue

        # 安全处理 NaN
        if np.isnan(raw_data).any():
            raw_data = np.nan_to_num(raw_data)

        # 动态计算当前指标的 X 轴 (解决展平后数据长度变化的问题)
        # 将展平后的总步数均匀映射到当前的 Episode 进度上
        x_axis = np.linspace(0, current_episode, len(raw_data))

        # A. 绘制原始数据 (浅色，透明度高)
        ax.plot(x_axis, raw_data, alpha=0.3, color='gray', label='Raw')

        # B. 绘制平滑数据 (深色，主趋势)
        # 根据数据长度动态调整平滑窗口，防止初期数据太少报错
        real_window = min(window_size, len(raw_data)) if len(raw_data) > 1 else 1
        smoothed_data = smooth_data(raw_data, real_window)

        # 使用不同的颜色循环
        line_color = plt.cm.tab10(i % 10)
        ax.plot(x_axis, smoothed_data, linewidth=2, color=line_color, label=f'Smooth (w={real_window})')

        # C. 装饰图表
        ax.set_title(key.replace('_', ' ').title())
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 只在第一个图显示图例，避免遮挡
        if i == 0:
            ax.legend(loc='best', fontsize='small')

    # 3. 删除多余的空子图
    for j in range(num_metrics, len(axes)):
        fig.delaxes(axes[j])

    # 4. 保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出标题空间

    save_path = os.path.join(RESULT_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=100)

    plt.close(fig)  # 极其重要：关闭图像释放内存
    print(f"[Plotter] Curves updated at {save_path}")