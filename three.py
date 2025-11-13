import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 假设你的文件名为：
file_paths = {
    'metis': 'metis_results_detailed_notDetected.csv',
    'mpc': 'mpc_results_detailed_new_829.csv',
    'sandpiper': 'sandpiper_results_detailed_notDetected.csv',
    'random': 'random_results_detailed.csv'
}

# 绘制的指标列表，及其对应的标题
plot_metrics = {
    # 'cost': 'Cost',  # 新增的指标
    'resource_contention': 'Resource Contention',
    'cohabitation_time_cumulative': 'Cumulative Cohabitation Time',
    'cpu_utilization_instant': 'Instant CPU Utilization',
    'memory_utilization_instant': 'Instant Memory Utilization',
    'cluster_risk': 'Cluster Risk',
    'migrations': 'Cumulative Migrations'  # 这里的migrations在绘图时会特殊处理
}

# 用于存储数据的字典
data_frames = {}
for name, path in file_paths.items():
    try:
        df = pd.read_csv(path)
        # 对 migrations 列进行累加操作
        df['migrations_cumulative'] = df['migrations'].cumsum()
        data_frames[name] = df
    except FileNotFoundError:
        print(f"Error: File not found at {path}. Please check the file path.")
        # 如果文件不存在，则不继续执行
        data_frames = {}
        break
    except KeyError as e:
        print(f"Error: Column {e} not found in {path}. Please check the CSV file content.")
        data_frames = {}
        break

# 如果数据加载成功，则开始绘图
if data_frames:
    # 调整为2x3的子图布局以容纳7个图
    # 7个图只需要3x3布局中的7个位置
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 调整figsize以适应更大的布局
    # 将axes转换为一维数组，方便循环
    axes = axes.flatten()

    # 定义颜色和标签
    colors = ['r', 'g', 'b', 'y']
    labels = list(data_frames.keys())

    # 循环绘制每一张图
    for i, (metric, title) in enumerate(plot_metrics.items()):
        ax = axes[i]

        # 定义纵坐标标签，默认为标题
        y_label = title

        # 遍历三套方案的数据
        for j, (name, df) in enumerate(data_frames.items()):
            # 横坐标为第一列数据 (window_id)
            x = df.iloc[:, 0]

            # 根据 metric 名称选择纵坐标数据
            y_col = metric
            if metric == 'migrations':
                y_col = 'migrations_cumulative'  # 使用累加后的数据

            # 检查列是否存在
            if y_col not in df.columns:
                print(f"Warning: Column '{y_col}' not found in {name}'s data. Skipping.")
                continue

            y = df[y_col]

            # --- 修改部分开始 ---
            # 根据不同的指标调整数据和纵坐标标签
            if metric == 'cpu_utilization_instant':
                y = y / 10
                y_label = f"{title} (%)"
            elif metric == 'memory_utilization_instant':
                y = y / 50
                y_label = f"{title} (%)"
            # --- 修改部分结束 ---

            # 绘制曲线
            ax.plot(x, y, color=colors[j], label=labels[j])

        # 设置图表标题和标签
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Window ID', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)  # 使用上面定义的y_label
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)

    # 由于只有6个子图，2x3布局没有空位
    # 如果未来增加图表，此部分需要调整
    # 隐藏多余的子图
    # for i in range(len(plot_metrics), len(axes)):
    #     fig.delaxes(axes[i])

    # 调整布局，使子图之间不重叠
    plt.tight_layout()

    # 定义保存路径和文件名
    save_path = 'combined_results_plot_with_cost_new_829.png'

    # 保存图表为PNG文件，dpi=300可以保证清晰度
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已成功保存至：{save_path}")

    # 显示图表
    plt.show()