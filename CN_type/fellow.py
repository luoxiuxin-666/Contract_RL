import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 基础配置
# ==========================================
SAVE_FOLDER = 'paper_simulation_two_graphs'
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# 设定 5 个具体的算力节点 (焦耳)
nodes_theta = np.array([5500, 7000, 10000, 14000, 18000])
node_names = ['N1', 'N2', 'N3', 'N4', 'N5']

# 实时计算标准差 (Sigma)
current_sigma = np.std(nodes_theta)
print(f"当前批次标准差 Sigma: {current_sigma:.2f}")

# 生成平滑曲线数据
x_smooth = np.linspace(0, 90000, 1000)

# 定义 k 值配置
configs = [
    (0.5, '#2ca02c', '--', r'$k=0.5$ (Low Sensitivity)'),  # 绿色
    (1.5, '#1f77b4', '-', r'$k=1.5$ (Balanced)'),  # 蓝色
    (5.0, '#d62728', '-.', r'$k=5.0$ (High Sensitivity)')  # 红色
]


# ==========================================
# 2. 通用绘图函数 (避免代码重复)
# ==========================================
def plot_and_save(data_func, title, ylabel, filename, y_limit, text_offset_base):
    plt.figure(figsize=(14, 8))

    # --- A. 画 5 条固定的垂直参考线 (背景) ---
    for x, name in zip(nodes_theta, node_names):
        plt.vlines(x, y_limit[0], y_limit[1], color='gray', linestyle=':', alpha=0.4, linewidth=1.5)
        # 标记节点名字 (放在X轴附近)
        plt.text(x, y_limit[0] + 0.01, name, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

    # --- B. 循环绘制每一条曲线 ---
    for k, color, style, label in configs:
        # 1. 计算曲线数据
        y_smooth = data_func(x_smooth, k, current_sigma)
        plt.plot(x_smooth, y_smooth, color=color, linestyle=style, linewidth=2.5, label=label, alpha=0.9)

        # 2. 计算这5个点的具体值
        y_nodes = data_func(nodes_theta, k, current_sigma)

        # 3. 画散点
        plt.scatter(nodes_theta, y_nodes, color=color, s=80, zorder=10, edgecolor='white', linewidth=1.5)

        # 4. 标注数值
        for x, y in zip(nodes_theta, y_nodes):
            # 动态调整位置
            offset = text_offset_base
            va = 'bottom'

            # 特殊处理：防止文字挤出边界或重叠
            if k == 5.0 and y > (y_limit[1] - 0.1):  # 接近顶部时，字往下放
                offset = (0, -15)
                va = 'top'
            elif k == 0.5:  # 绿色线通常在中间，往下放一点避开蓝色
                offset = (0, -15)
                va = 'top'

            # 如果是成本图(曲线下降)，红色线在低能耗区很高，也要调整
            if 'Cost' in title and k == 5.0 and y > 1.8:
                offset = (15, 0)  # 往右偏一点
                va = 'center'

            plt.annotate(f'{y:.2f}',
                         xy=(x, y),
                         xytext=offset,
                         textcoords='offset points',
                         ha='center',
                         va=va,
                         fontsize=9,
                         color=color,
                         fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85, lw=0.5))

    # --- C. 图表修饰 ---
    plt.title(f'{title}\nBased on Real Batch $\sigma={current_sigma:.0f}$', fontsize=16)
    plt.xlabel('Absolute Energy Margin (Joule)', fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.ylim(y_limit)
    plt.xlim(-2000, 92000)
    plt.grid(True, linestyle='-', alpha=0.1)
    plt.legend(fontsize=12, frameon=True, shadow=True)

    # 保存
    save_path = os.path.join(SAVE_FOLDER, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存: {save_path}")
    plt.show()


# ==========================================
# 3. 执行绘图 - 图一：喜好程度 (y)
# ==========================================
def func_preference(theta, k, sigma):
    return 1 / (1 + np.exp(-k * (theta / sigma)))


plot_and_save(
    data_func=func_preference,
    title='Figure A: Normalized Preference Scores ($y$)',
    ylabel='Score $y \\in (0.5, 1.0)$',
    filename='Preference_Score_y.png',
    y_limit=(0.45, 1.05),
    text_offset_base=(0, 10)
)


# ==========================================
# 4. 执行绘图 - 图二：成本惩罚 (1/y)
# ==========================================
def func_cost(theta, k, sigma):
    return 1 + np.exp(-k * (theta / sigma))


plot_and_save(
    data_func=func_cost,
    title='Figure B: Cost Penalty Multiplier ($1/y$)',
    ylabel='Multiplier $1/y \\in (1.0, 2.0)$',
    filename='Cost_Penalty_1_over_y.png',
    y_limit=(0.95, 2.05),
    text_offset_base=(0, 10)
)