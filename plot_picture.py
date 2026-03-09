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

    # 2. 遍历绘制每个指标
    keys = list(metrics_dict.keys())

    # 计算 X 轴刻度
    # 假设所有指标的记录频率是一样的，取第一个非空列表计算步长
    any_key = keys[0]
    data_len = len(metrics_dict[any_key])
    if data_len == 0: return

    # 计算记录间隔 (Log Interval)
    step_interval = current_episode / data_len
    x_axis = np.arange(1, data_len + 1) * step_interval

    for i, key in enumerate(keys):
        ax = axes[i]
        raw_data = np.array(metrics_dict[key]).flatten()

        # 检查是否为空或含 NaN
        if len(raw_data) == 0:continue
        if np.isnan(raw_data).any():
            # 可选：填充 NaN
            raw_data = np.nan_to_num(raw_data)

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

    # 额外保存一份带时间戳的，防止覆盖后想找回历史
    # import time
    # ts_path = os.path.join(RESULT_DIR, f'history/curves_{int(time.time())}.png')
    # os.makedirs(os.path.dirname(ts_path), exist_ok=True)
    # plt.savefig(ts_path, dpi=100)

    plt.close(fig)  # 极其重要：关闭图像释放内存
    print(f"[Plotter] Curves updated at {save_path}")