import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 假设你的文件名为：
file_paths = {
    'metis': 'metis_results_detailed_20_Services.csv',
    'sandpiper': 'sandpiper_results_detailed_20_Services.csv',
    'random': 'random_results_detailed_20_Services.csv',
    'mpc': 'mpc_results_detailed_20_Services.csv',
}

# 绘制的指标列表，及其对应的标题
plot_metrics = {
    # 'cost': 'Cost',  # 新增的指标
    'resource_contention': 'Resource Contention',
    'cohabitation_time_cumulative': 'Cumulative Cohabitation Time',
    'cpu_utilization_instant': 'Instant CPU Utilization',
    # 'memory_utilization_instant': 'Instant Memory Utilization',
    'cluster_risk': 'Cluster Risk',
    # 'migrations': 'Cumulative Migrations'  # 这里的migrations在绘图时会特殊处理
}


# 颜色列表（从下往上：蓝、红、绿、紫、浅蓝）
colors = [
    '#4f81bd',  # 蓝
    '#c0504d',  # 红
    '#9bbb59',  # 绿
    '#8064a2',  # 紫
    '#4bacc6',  # 浅蓝
]

# 标记图案列表（每个方案用不同形状）
markers = ['o', 's', '^', 'D', 'v']  # 圆、方、三角上、菱形、三角下

# 用于存储数据的字典
data_frames = {}
for name, path in file_paths.items():
    try:
        df = pd.read_csv(path)
        if 'migrations' in df.columns:
            df['migrations_cumulative'] = df['migrations'].cumsum()
        data_frames[name] = df[df.iloc[:, 0]<=200]  # 过滤 window_id 小于等于 200 的数据
    except FileNotFoundError:
        print(f"Error: File not found at {path}. Please check the file path.")
        data_frames = {}
        break
    except KeyError as e:
        print(f"Error: Column {e} not found in {path}. Please check the CSV file content.")
        data_frames = {}
        break

# 如果数据加载成功，则开始绘图
if data_frames:
    # 创建 1x4 子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()

    # 定义标签顺序
    names = list(data_frames.keys())
    colors_used = [colors[i % len(colors)] for i in range(len(names))]
    markers_used = [markers[i % len(markers)] for i in range(len(names))]

    # 循环绘制每一张图
    for i, (metric, title) in enumerate(plot_metrics.items()):
        ax = axes[i]
        y_label = title

        # 遍历每个方案的数据
        for j, (name, df) in enumerate(data_frames.items()):
            x = df.iloc[:, 0]  # window_id
            y_col = metric
            if metric == 'migrations':
                y_col = 'migrations_cumulative'

            if y_col not in df.columns:
                print(f"Warning: Column '{y_col}' not found in {name}'s data. Skipping.")
                continue

            y = df[y_col]

            # --- 关键：取 7 个点，按 1/7, 2/7... 分布 ---
            n_points = len(x)
            step = max(1, n_points // 8)  # 每隔 step 个点取一个
            indices = np.arange(0, n_points, step)[:9]  # 最多取 7 个点

            # 根据指标调整数据
            if metric == 'cpu_utilization_instant':
                y = y / 2
                y_label = f"{title} (%)"
            elif metric == 'memory_utilization_instant':
                y = y / 50
                y_label = f"{title} (%)"

            # 提取对应的 x 和 y 值
            x_sample = x.iloc[indices].values
            y_sample = y.iloc[indices].values
            # --- 绘制：先画线，再加点 ---
            # 画线并设置 label（关键！）
            ax.plot(x, y, color=colors_used[j], linewidth=1.5, linestyle='-', label=name)

            # 画点但不设 label
            ax.scatter(x_sample, y_sample,
                       color=colors_used[j],
                       marker=markers_used[j],
                       s=50,
                       edgecolors='black', linewidth=0.5,
                       zorder=5)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Window ID', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)

    # 隐藏多余的子图
    for k in range(len(plot_metrics), 4):
        fig.delaxes(axes[k])

    plt.tight_layout()

    save_path = 'comparation_results_10_with_points.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已成功保存至：{save_path}")

    plt.show()

# # 2.用于存储数据的字典
# data_frames = {}
# for name, path in file_paths.items():
#     try:
#         df = pd.read_csv(path)
#         # 对 migrations 列进行累加操作（如果需要）
#         if 'migrations' in df.columns:
#             df['migrations_cumulative'] = df['migrations'].cumsum()
#         data_frames[name] = df
#     except FileNotFoundError:
#         print(f"Error: File not found at {path}. Please check the file path.")
#         data_frames = {}
#         break
#     except KeyError as e:
#         print(f"Error: Column {e} not found in {path}. Please check the CSV file content.")
#         data_frames = {}
#         break
# if data_frames:
#     # 改为 1 行 4 列子图
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 宽度拉长，高度适当减小
#     # axes 已经是一维数组（因为是 1 行），无需 flatten；但为了兼容性也可以保留
#     if len(plot_metrics) == 1:
#         axes = [axes]  # 防止单图时 axes 不是列表
#     else:
#         axes = axes.flatten() if hasattr(axes, 'flatten') else axes
#
#     # 定义颜色和标签
#     colors = ['r', 'y', 'b', 'g']
#     labels = list(data_frames.keys())
#
#     # 循环绘制每一张图（最多4个）
#     for i, (metric, title) in enumerate(plot_metrics.items()):
#         if i >= 4:  # 只画前4个，防止越界
#             break
#         ax = axes[i]
#
#         y_label = title
#
#         for j, (name, df) in enumerate(data_frames.items()):
#             x = df.iloc[:, 0]  # window_id
#
#             y_col = metric
#             if metric == 'migrations':
#                 y_col = 'migrations_cumulative'
#
#             if y_col not in df.columns:
#                 print(f"Warning: Column '{y_col}' not found in {name}'s data. Skipping.")
#                 continue
#
#             y = df[y_col]
#
#             # 根据指标调整数据和标签
#             if metric == 'cpu_utilization_instant':
#                 y = y / 2
#                 y_label = f"{title} (%)"
#             elif metric == 'memory_utilization_instant':
#                 y = y / 50
#                 y_label = f"{title} (%)"
#
#             ax.plot(x, y, color=colors[j], label=labels[j])
#
#         ax.set_title(title, fontsize=14)
#         ax.set_xlabel('Time Step', fontsize=12)
#         ax.set_ylabel(y_label, fontsize=12)
#         ax.legend(loc='best')
#         ax.grid(True, linestyle='--', alpha=0.6)
#
#     # 隐藏多余的子图（如果 plot_metrics 少于4个）
#     for k in range(len(plot_metrics), 4):
#         fig.delaxes(axes[k])
#
#     plt.tight_layout()
#
#     save_path = 'comparation_results_20_Services'
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"图表已成功保存至：{save_path}")
#
#     plt.show()

# ##迁移次数画图（柱状图）
# row_indices = list(range(100, 800, 100))  # [0, 100, ..., 700]
#
# data_dict = {}
# colors = ['#4f81bd', '#c0504d', '#9bbb59', '#8064a2']  # 对应蓝、红、绿、紫
#
# for idx, (algo_name, file_path) in enumerate(file_paths.items()):
#     try:
#         df = pd.read_csv(file_path)
#         if len(df) < 701:
#             print(f"警告: {algo_name} 的数据不足 701 行，实际只有 {len(df)} 行。")
#             valid_indices = [i for i in row_indices if i < len(df)]
#         else:
#             valid_indices = row_indices
#
#         values = df.loc[valid_indices, 'migrations_cumulative'].values
#         data_dict[algo_name] = values
#     except Exception as e:
#         print(f"读取或处理 {algo_name} 文件时出错: {e}")
#         data_dict[algo_name] = np.full(len(row_indices), np.nan)
#
# x_labels = [str(i) for i in row_indices]
# x_pos = np.arange(len(x_labels))
#
# fig, ax = plt.subplots(figsize=(12, 6))
#
# bar_width = 0.2
# offsets = np.linspace(-1.5, 1.5, len(data_dict)) * bar_width
#
# for idx, (algo_name, values) in enumerate(data_dict.items()):
#     ax.bar(x_pos + offsets[idx], values, width=bar_width, label=algo_name, color=colors[idx])
#
# ax.set_xlabel('Time Step')
# ax.set_ylabel('Migrations')
# ax.set_title('Cumulative Migrations')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(x_labels)
# ax.legend()
# ax.grid(axis='y', linestyle='--', alpha=0.7)
#
# plt.tight_layout()
#
# # 保存图片
# plt.savefig('cumulative_migrations_20_services.png')
#
# plt.show()