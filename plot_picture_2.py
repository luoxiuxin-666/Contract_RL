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
    padding = data[:window_size - 1]
    return np.concatenate([padding, smoothed])


def preprocess_metrics(metrics_dict, current_episode, window_size):
    """
    将原始字典数据处理成绘图所需的 x 轴和平滑后的 y 轴数据
    """
    processed_data = {}

    for key, data_list in metrics_dict.items():
        if not data_list:
            continue

        # 1. 统一提取数据 (安全处理 tensor.item() 和嵌套列表)
        cleaned_list = []
        for item in data_list:
            if isinstance(item, (list, tuple, np.ndarray)):
                cleaned_list.append([x.item() if hasattr(x, 'item') else float(x) for x in item])
            else:
                cleaned_list.append([item.item() if hasattr(item, 'item') else float(item)])

        # 2. 寻找当前指标数据中最长的子序列长度
        lengths = [len(x) for x in cleaned_list]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            continue

        # 3. 使用最后一个元素对齐补全短数据
        padded_data = []
        for item_list in cleaned_list:
            if len(item_list) == 0:
                padded_data.append([np.nan] * max_len)
            elif len(item_list) < max_len:
                last_element = item_list[-1]
                padding_length = max_len - len(item_list)
                padded_data.append(item_list + [last_element] * padding_length)
            else:
                padded_data.append(item_list)

        # 4. 转换为规范的 float 数组并展平
        raw_data = np.array(padded_data, dtype=np.float64).flatten()

        if len(raw_data) == 0:
            continue

        # 安全处理 NaN
        if np.isnan(raw_data).any():
            raw_data = np.nan_to_num(raw_data)

        # 5. 动态计算当前指标的 X 轴
        x_axis = np.linspace(0, current_episode, len(raw_data))

        # 6. 计算平滑数据
        real_window = min(window_size, len(raw_data)) if len(raw_data) > 1 else 1
        smoothed_y = smooth_data(raw_data, real_window)

        processed_data[key] = {
            'x': x_axis,
            'y': smoothed_y,
            'window': real_window
        }

    return processed_data


def plot_learning_curves_(metrics_dict, current_episode,picture_name = None, window_size=20, combine_plots=False):
    """
    改进版绘图函数：
    1. 去除了背景的灰色 Raw 数据，只保留 Smooth 曲线
    2. combine_plots = False: 每个数据单独存为一张独立图片 (无标签/图例)
    3. combine_plots = True: 所有数据画在【同一个坐标系】里做对比，并带有各自的名称标签
    """
    if not metrics_dict:
        return

    # 获取预处理后可直接绘制的数据
    processed_data = preprocess_metrics(metrics_dict, current_episode, window_size)
    if not processed_data:
        return

    keys = list(processed_data.keys())
    num_metrics = len(keys)

    # ================= 模式1：联合展示 (所有数据画在【同一个坐标系】内对比) =================
    if combine_plots:
        # 只创建单个图表和单个坐标系
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Training Metrics Comparison (Episode {current_episode})', fontsize=16)

        # 遍历所有数据，画在同一个 ax 上
        for i, key in enumerate(keys):
            data = processed_data[key]

            color_palette = ['red', 'blue', 'green', 'darkorange', 'purple',
                             'brown', 'magenta', 'teal', 'olive', 'black']

            # 按顺序从列表中取出颜色（如果线条超过10条，会自动循环利用）
            line_color = color_palette[i % len(color_palette)]

            # 【修改点】：直接使用 key 作为标签，原样输出，不做任何大小写转换
            metric_name = key
            #
            # # 标签设为数据的名称
            # metric_name = key.replace('_', ' ').title()

            # 绘制平滑曲线，附加标签用于生成图例
            ax.plot(data['x'], data['y'], linewidth=2, color=line_color, label=metric_name)

        # 设置统一的坐标轴信息
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 显示图例，里面包含所有画在这张图上的数据名称
        ax.legend(loc='best', fontsize='medium')
        name = picture_name if picture_name else 'training_curves_combined'
        name += '.png'
        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, name)
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"[Plotter] Combined comparison curve updated at {save_path}")

    # ================= 模式2：单独展示 (每个数据独立存为一张图片) =================
    else:
        for i, key in enumerate(keys):
            # 每个数据创建一张独立的图表
            fig, ax = plt.subplots(figsize=(6, 4))
            data = processed_data[key]

            line_color = plt.cm.tab10(i % 10)
            # metric_name = key.replace('_', ' ').title()
            metric_name = key
            # 单独展示时不使用 label，不生成图例
            ax.plot(data['x'], data['y'], linewidth=2.5, color=line_color)

            ax.set_title(f"{metric_name} (Episode {current_episode})")
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()

            # 过滤掉文件名中的非法字符
            safe_filename = key.replace('/', '_').replace(' ', '_') + '.png'
            save_path = os.path.join(RESULT_DIR, safe_filename)
            plt.savefig(save_path, dpi=100)
            plt.close(fig)  # 释放内存

        print(f"[Plotter] {num_metrics} individual curve images updated in {RESULT_DIR}")