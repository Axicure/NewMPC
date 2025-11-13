import os
import time
import numpy as np
import pandas as pd
import gc
import logging
from data_loader import DataLoader
from prediction_models import PredictionModels
from mpc_controller import MPCController
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple

# 设置matplotlib全局字体为英文
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
# 禁用中文负号显示问题
mpl.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='mpc_scheduler.log',
    filemode='a'
)
logger = logging.getLogger('mpc_scheduler')


class MPCScheduler:
    def __init__(self, vm_count: int = 50, pm_count: int = 39):
        """
        初始化MPC调度器
        :param vm_count: 虚拟机数量
        :param pm_count: 物理机数量
        """
        self.vm_count = vm_count
        self.pm_count = pm_count

        # 初始化数据加载器
        self.data_loader = DataLoader(vm_count=vm_count, pm_count=pm_count)

        # 加载数据
        self.data_loader.load_all_data()

        # 初始化预测模型
        self.prediction_models = PredictionModels(self.data_loader)

        # 初始化MPC控制器
        self.mpc_controller = MPCController(self.data_loader, self.prediction_models)

        # 记录模型训练状态
        self.models_trained = False

        # 记录运行结果 - 扩展的数据收集
        self.costs_history = []  # 记录每个时间窗口的成本
        self.migrations_history = []  # 记录每个时间窗口的迁移次数
        self.attacker_count_history = []  # 记录每个时间窗口的攻击者数量

        # 新增的数据收集
        self.cohabitation_time_history = []  # 共居时长（累计）
        self.cohabitation_malicious_vm_count_history = []  # 共居恶意VM数量（瞬时）
        self.server_total_risk_history = []  # 服务器总风险度（瞬时）
        self.vm_average_risk_history = []  # 虚拟机平均风险值（瞬时）
        self.cpu_utilization_history = []  # CPU利用率（瞬时）
        self.memory_utilization_history = []  # 内存利用率（瞬时）
        self.cumulative_migrations_history = []  # 迁移虚拟机数量（累计）
        self.active_servers_history = []  # 开启服务器数量（瞬时）

        # 累计变量
        self.total_cohabitation_time = 0
        self.total_migrations = 0

        # CPU和内存使用率的累计值（用于计算平均值）
        self.cumulative_cpu_usage = 0
        self.cumulative_memory_usage = 0

    def ensure_models_trained(self, force_retrain: bool = False):
        """
        确保所有模型都已经训练完成
        :param force_retrain: 是否强制重新训练模型
        """
        if not self.models_trained or force_retrain:
            print("开始训练预测模型...")
            # 训练所有模型
            self.prediction_models.train_all_models(force_retrain=force_retrain)
            self.models_trained = True
            print("预测模型训练完成")

    def calculate_cohabitation_metrics(self, current_window: int) -> Tuple[int, float]:
        """
        计算共居相关指标
        :param current_window: 当前时间窗口
        :return: (共居恶意VM数量, 当前窗口共居时间增量)
        """
        cohabitation_malicious_count = 0
        cohabitation_time_increment = 0

        # 获取每台物理机上的虚拟机列表
        pm_to_vms = {}
        for vm_id, pm_id in self.data_loader.vm_pm_mapping.items():
            if pm_id not in pm_to_vms:
                pm_to_vms[pm_id] = []
            pm_to_vms[pm_id].append(vm_id)

        # 计算每台物理机的共居情况
        for pm_id, vm_list in pm_to_vms.items():
            attackers = [vm_id for vm_id in vm_list if self.mpc_controller.vm_is_attacker.get(vm_id, False)]
            normal_vms = [vm_id for vm_id in vm_list if not self.mpc_controller.vm_is_attacker.get(vm_id, False)]

            if attackers and normal_vms:
                # 有攻击者和正常VM共存
                cohabitation_malicious_count += len(attackers)
                # 共居时间增量 = 攻击者数量 × 正常VM数量 × 时间窗口长度（假设为1）
                cohabitation_time_increment += len(attackers) * len(normal_vms)

        return cohabitation_malicious_count, cohabitation_time_increment

    def calculate_server_total_risk(self) -> float:
        """
        计算服务器总风险度
        :return: 服务器总风险度
        """
        total_risk = 0
        active_pms = self.data_loader.get_active_physical_machines()

        for pm_id in active_pms:
            server_risk = self.mpc_controller.calculate_server_risk(pm_id)
            total_risk += server_risk

        return total_risk

    def calculate_vm_average_risk(self, current_window: int) -> float:
        """
        计算虚拟机平均风险值
        :param current_window: 当前时间窗口
        :return: 虚拟机平均风险值
        """
        window_data = self.data_loader.get_time_window_data(current_window)

        if not window_data:
            return 0.0

        total_risk = sum(data['risk_level'] for data in window_data.values())
        return total_risk / len(window_data)

    def calculate_resource_utilization(self, current_window: int) -> Tuple[float, float]:
        """
        计算CPU和内存利用率：按物理机聚合，计算其平均值
        :param current_window: 当前时间窗口
        :return: (平均CPU利用率, 平均内存利用率)
        """
        window_data = self.data_loader.get_time_window_data(current_window)

        if not window_data:
            print(f"警告: 时间窗口 {current_window} 没有数据")
            return 0.0, 0.0

        # 初始化每台物理机的资源使用统计
        pm_resource_usage: Dict[int, Dict[str, float]] = {}

        # 遍历所有VM，聚合到对应的PM
        for vm_id, vm_data in window_data.items():
            pm_id = self.data_loader.vm_pm_mapping.get(vm_id)
            if pm_id is None:
                continue

            if pm_id not in pm_resource_usage:
                pm_resource_usage[pm_id] = {"cpu": 0.0, "memory": 0.0}

            pm_resource_usage[pm_id]["cpu"] += vm_data.get("cpu_usage", 0.0)
            pm_resource_usage[pm_id]["memory"] += vm_data.get("memory_usage", 0.0)

        num_active_pms = len(pm_resource_usage)
        if num_active_pms == 0:
            return 0.0, 0.0

        total_cpu_usage = sum(pm["cpu"] for pm in pm_resource_usage.values())
        total_memory_usage = sum(pm["memory"] for pm in pm_resource_usage.values())

        avg_cpu_utilization = total_cpu_usage / num_active_pms
        avg_memory_utilization = total_memory_usage / num_active_pms

        # 打印调试信息
        if current_window % 10 == 0:
            print(f"时间窗口 {current_window}: 活跃PM数量={num_active_pms}, " +
                  f"平均CPU={avg_cpu_utilization:.2f}%, 平均内存={avg_memory_utilization:.2f}%")

        return avg_cpu_utilization, avg_memory_utilization

    def collect_metrics(self, current_window: int):
        """
        收集当前时间窗口的所有指标
        :param current_window: 当前时间窗口
        """
        # 计算共居指标
        cohabitation_malicious_count, cohabitation_time_increment = self.calculate_cohabitation_metrics(current_window)
        self.total_cohabitation_time += cohabitation_time_increment

        # 计算服务器总风险度
        server_total_risk = self.calculate_server_total_risk()

        # 计算虚拟机平均风险值
        vm_average_risk = self.calculate_vm_average_risk(current_window)

        # 计算资源利用率
        cpu_utilization, memory_utilization = self.calculate_resource_utilization(current_window)
        self.cumulative_cpu_usage += cpu_utilization
        self.cumulative_memory_usage += memory_utilization

        # 计算活跃服务器数量
        active_servers = len(self.data_loader.get_active_physical_machines())

        # 更新累计迁移次数
        current_migrations = len(self.mpc_controller.migrations_executed)
        self.total_migrations += current_migrations

        # 记录所有指标
        self.cohabitation_time_history.append(self.total_cohabitation_time)
        self.cohabitation_malicious_vm_count_history.append(cohabitation_malicious_count)
        self.server_total_risk_history.append(server_total_risk)
        self.vm_average_risk_history.append(vm_average_risk)
        self.cpu_utilization_history.append(cpu_utilization)
        self.memory_utilization_history.append(memory_utilization)
        self.cumulative_migrations_history.append(self.total_migrations)
        self.active_servers_history.append(active_servers)

    def run_mpc_optimization(self, start_window: int = 0, end_window: int = None, prediction_steps: int = 3):
        """
        运行MPC优化
        :param start_window: 起始时间窗口
        :param end_window: 结束时间窗口，如果为None则运行到最后一个窗口
        :param prediction_steps: 预测步数
        """
        # 确保模型已训练
        self.ensure_models_trained()

        # 确定时间窗口范围
        if end_window is None:
            # 获取第一个虚拟机的时间窗口数量作为总窗口数
            vm_id = next(iter(self.data_loader.time_windows.keys()))
            end_window = len(self.data_loader.time_windows[vm_id]) - 1

        # 检查所有VM的时间窗口数量是否一致
        min_windows = float('inf')
        max_windows = 0
        for vm_id, windows in self.data_loader.time_windows.items():
            window_count = len(windows)
            min_windows = min(min_windows, window_count)
            max_windows = max(max_windows, window_count)

        print(f"VM时间窗口统计: 最少={min_windows}, 最多={max_windows}")

        # 使用最小窗口数量作为安全的结束窗口
        if end_window is None:
            end_window = min_windows - 1
        else:
            end_window = min(end_window, min_windows - 1)

        print(f"调整后的时间窗口范围: {start_window} - {end_window}")

        print(f"开始MPC优化，时间窗口范围: {start_window} - {end_window}")

        # 记录开始时间
        start_time = time.time()

        # 对每个时间窗口进行优化
        for current_window in range(start_window, end_window + 1):
            print(f"\n处理时间窗口 {current_window}/{end_window}")

            # 获取当前窗口的数据
            current_window_data = self.data_loader.get_time_window_data(current_window)

            # 预测未来数据
            future_data = {}

            try:
                # 获取预测数据
                if current_window + prediction_steps <= end_window:
                    # 使用滑动窗口预测未来数据
                    print(f"预测未来 {prediction_steps} 个时间窗口的数据")
                    predicted_data = self.prediction_models.predict_future_data(current_window)

                    # 将预测数据与当前时间窗口数据合并
                    for vm_id in current_window_data.keys():
                        future_data[vm_id] = [current_window_data[vm_id]]
                        if vm_id in predicted_data:
                            future_data[vm_id].extend(predicted_data[vm_id])
                else:
                    # 如果到了结尾，使用实际数据
                    for window_id in range(current_window, min(current_window + prediction_steps + 1, end_window + 1)):
                        window_data = self.data_loader.get_time_window_data(window_id)
                        for vm_id, data in window_data.items():
                            if vm_id not in future_data:
                                future_data[vm_id] = []
                            future_data[vm_id].append(data)

            except Exception as e:
                logger.error(f"预测未来数据时出错: {str(e)}")
                # 使用最后一个时间窗口的数据作为未来数据
                for vm_id, data in current_window_data.items():
                    future_data[vm_id] = [data] * (prediction_steps + 1)

            # 在优化前保存当前虚拟机-物理机映射
            vm_pm_mapping_before = self.data_loader.vm_pm_mapping.copy()

            # 记录优化前的成本
            cost_before = self.mpc_controller.calculate_cost(
                vm_pm_mapping_before,
                vm_pm_mapping_before,
                current_window_data
            )

            # 优化虚拟机放置
            try:
                # 执行MPC优化
                optimized_mapping = self.mpc_controller.optimize_vm_placement(current_window)

                # 获取优化后的成本（避免重复计算）
                if hasattr(self.mpc_controller, 'last_optimization_cost'):
                    cost_after = self.mpc_controller.last_optimization_cost
                else:
                    # 如果没有存储优化后的成本，重新计算
                    cost_after = self.mpc_controller.calculate_cost(
                        optimized_mapping,
                        vm_pm_mapping_before,
                        current_window_data
                    )

                # 记录攻击者数量
                attacker_count = sum(1 for is_attacker in self.mpc_controller.vm_is_attacker.values() if is_attacker)

                # 记录迁移次数 - 使用控制器中记录的实际迁移
                migrations = len(self.mpc_controller.migrations_executed)

                # 更新历史记录
                self.costs_history.append(cost_after)
                self.migrations_history.append(migrations)
                self.attacker_count_history.append(attacker_count)

                # 记录成本组成部分的历史
                if not hasattr(self, 'cost_components_history'):
                    self.cost_components_history = []

                self.cost_components_history.append(self.mpc_controller.last_cost_components.copy())

                # 收集新增的指标
                self.collect_metrics(current_window)

                print(f"时间窗口 {current_window} 优化完成: 当前成本 {cost_before:.2f}" +
                      (f" -> 优化后成本 {cost_after:.2f}" if migrations > 0 else f" (无迁移, 成本不变)") +
                      f", 迁移次数: {migrations}, 攻击者数量: {attacker_count}")

            except Exception as e:
                logger.error(f"优化时间窗口 {current_window} 时出错: {str(e)}")
                print(f"优化时间窗口 {current_window} 时出错: {str(e)}")

                # 记录错误情况
                self.costs_history.append(cost_before)
                self.migrations_history.append(0)
                self.attacker_count_history.append(
                    sum(1 for is_attacker in self.mpc_controller.vm_is_attacker.values() if is_attacker))

                if not hasattr(self, 'cost_components_history'):
                    self.cost_components_history = []

                # 添加空的成本组成记录
                self.cost_components_history.append({
                    'cluster_risk': 0.0,
                    'power_consumption': 0.0,
                    'migration_cost': 0.0,
                    'resource_contention': 0.0,
                    'total_cost': cost_before
                })

                # 收集指标（即使出错也要收集）
                self.collect_metrics(current_window)

            # 每10个时间窗口清理一次内存
            if current_window % 10 == 0:
                gc.collect()

        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time

        print(f"\nMPC优化完成，总耗时: {duration:.2f}秒")
        print(f"处理了 {end_window - start_window + 1} 个时间窗口")
        print(f"总迁移次数: {sum(self.migrations_history)}")

        # 清理内存
        gc.collect()

    def plot_time_series_charts(self, save_path: str = None):
        """
        绘制时间序列折线图：共居时长、CPU使用率、内存使用率、迁移次数
        :param save_path: 保存图表的路径
        """
        if not self.cohabitation_time_history:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

        time_windows = list(range(len(self.cohabitation_time_history)))

        # 共居时长折线图
        axes[0, 0].plot(time_windows, self.cohabitation_time_history, 'r-', linewidth=2)
        axes[0, 0].set_title('Cohabitation Time (Cumulative)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Cohabitation Time')
        axes[0, 0].grid(True, alpha=0.3)

        # CPU使用率折线图
        axes[0, 1].plot(time_windows, self.cpu_utilization_history, 'b-', linewidth=2)
        axes[0, 1].set_title('CPU Utilization', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('CPU Utilization (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # 内存使用率折线图
        axes[1, 0].plot(time_windows, self.memory_utilization_history, 'g-', linewidth=2)
        axes[1, 0].set_title('Memory Utilization', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Memory Utilization (%)')
        axes[1, 0].set_xlabel('Time Window')
        axes[1, 0].grid(True, alpha=0.3)

        # 累计迁移次数折线图
        axes[1, 1].plot(time_windows, self.cumulative_migrations_history, 'purple', linewidth=2)
        axes[1, 1].set_title('Cumulative Migrations', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Migration Count')
        axes[1, 1].set_xlabel('Time Window')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time series charts saved to {save_path}")
        else:
            plt.show()

    def plot_risk_comparison_charts(self, save_path: str = None):
        """
        绘制VM风险值、服务器总风险度-不同物理机数量的柱状图
        :param save_path: 保存图表的路径
        """
        if not self.vm_average_risk_history:
            print("No data to plot")
            return

        # 创建不同物理机数量的数据（这里使用活跃服务器数量作为代理）
        unique_server_counts = sorted(list(set(self.active_servers_history)))

        vm_risk_by_servers = {}
        server_risk_by_servers = {}

        for server_count in unique_server_counts:
            vm_risks = []
            server_risks = []

            for i, active_count in enumerate(self.active_servers_history):
                if active_count == server_count:
                    vm_risks.append(self.vm_average_risk_history[i])
                    server_risks.append(self.server_total_risk_history[i])

            if vm_risks:
                vm_risk_by_servers[server_count] = np.mean(vm_risks)
                server_risk_by_servers[server_count] = np.mean(server_risks)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # VM平均风险值柱状图
        server_counts = list(vm_risk_by_servers.keys())
        vm_risks = list(vm_risk_by_servers.values())

        bars1 = ax1.bar(server_counts, vm_risks, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Average VM Risk by Server Count', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Number of Active Servers')
        ax1.set_ylabel('Average VM Risk Value')
        ax1.grid(True, alpha=0.3)

        # 在柱状图上显示数值
        for bar, risk in zip(bars1, vm_risks):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{risk:.2f}', ha='center', va='bottom')

        # 服务器总风险度柱状图
        server_risks = list(server_risk_by_servers.values())

        bars2 = ax2.bar(server_counts, server_risks, color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.set_title('Total Server Risk by Server Count', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Active Servers')
        ax2.set_ylabel('Total Server Risk')
        ax2.grid(True, alpha=0.3)

        # 在柱状图上显示数值
        for bar, risk in zip(bars2, server_risks):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{risk:.1f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Risk comparison charts saved to {save_path}")
        else:
            plt.show()

    def plot_results(self, save_path: str = None):
        """
        绘制运行结果
        :param save_path: 保存图表的路径，如果为None则显示图表
        """
        if not self.costs_history:
            print("No data to plot")
            return

        # 创建包含4个子图的图表
        fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

        # 绘制成本历史
        axes[0].plot(self.costs_history, 'b-', label='Total Cost')
        axes[0].set_ylabel('Cost')
        axes[0].set_title('MPC Optimization Results')
        axes[0].legend()
        axes[0].grid(True)

        # 绘制成本组成部分
        if hasattr(self, 'cost_components_history') and self.cost_components_history:
            cluster_risk = [history.get('cluster_risk', 0.0) for history in self.cost_components_history]
            power_consumption = [history.get('power_consumption', 0.0) for history in self.cost_components_history]
            migration_cost = [history.get('migration_cost', 0.0) for history in self.cost_components_history]
            resource_contention = [history.get('resource_contention', 0.0) for history in self.cost_components_history]

            axes[1].plot(cluster_risk, 'r-', label='Cohabitation Risk')
            axes[1].plot(power_consumption, 'g-', label='Power Consumption')
            axes[1].plot(migration_cost, 'y-', label='Migration Cost')
            axes[1].plot(resource_contention, 'm-', label='Resource Contention')
            axes[1].set_ylabel('Cost Components')
            axes[1].legend()
            axes[1].grid(True)

        # 绘制迁移次数历史
        axes[2].bar(range(len(self.migrations_history)), self.migrations_history, color='g', label='Migration Count')
        axes[2].set_ylabel('Migrations')
        axes[2].legend()
        axes[2].grid(True)

        # 绘制攻击者数量历史
        axes[3].plot(self.attacker_count_history, 'r-', label='Attackers Count')
        axes[3].set_ylabel('Attackers')
        axes[3].legend()
        axes[3].grid(True)

        # 绘制累计迁移次数
        cumulative_migrations = np.cumsum(self.migrations_history)
        axes[4].plot(cumulative_migrations, 'k-', label='Cumulative Migrations')
        axes[4].set_xlabel('Time Window')
        axes[4].set_ylabel('Cumulative Migrations')
        axes[4].legend()
        axes[4].grid(True)

        # 调整布局
        plt.tight_layout()

        # 保存或显示图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results chart saved to {save_path}")
        else:
            plt.show()

    def export_results(self, file_path: str = "mpc_results.csv"):
        """
        导出运行结果到CSV文件 - 包含所有新增指标
        :param file_path: CSV文件路径
        """
        if not self.costs_history:
            print("No data to export")
            return

        # 计算平均CPU和内存利用率（0-t时间的平均值）
        num_windows = len(self.cpu_utilization_history)
        avg_cpu_utilization = []
        avg_memory_utilization = []

        for i in range(num_windows):
            avg_cpu = sum(self.cpu_utilization_history[:i + 1]) / (i + 1)
            avg_memory = sum(self.memory_utilization_history[:i + 1]) / (i + 1)
            avg_cpu_utilization.append(avg_cpu)
            avg_memory_utilization.append(avg_memory)

        # 创建结果DataFrame
        results = pd.DataFrame({
            'window_id': list(range(len(self.costs_history))),
            'cost': self.costs_history,
            'migrations': self.migrations_history,
            'attacker_count': self.attacker_count_history,

            # 新增的七个指标
            'cohabitation_time_cumulative': self.cohabitation_time_history,  # 共居时长（累计）
            'cohabitation_malicious_vm_count': self.cohabitation_malicious_vm_count_history,  # 共居恶意VM数量（瞬时）
            'server_total_risk': self.server_total_risk_history,  # 服务器总风险度（瞬时）
            'vm_average_risk': self.vm_average_risk_history,  # 虚拟机平均风险值（瞬时）
            'cpu_utilization_avg': avg_cpu_utilization,  # CPU利用率（0-t平均值）
            'memory_utilization_avg': avg_memory_utilization,  # 内存利用率（0-t平均值）
            'migrations_cumulative': self.cumulative_migrations_history,  # 迁移虚拟机数量（累计）
            'active_servers_count': self.active_servers_history,  # 开启服务器数量（瞬时）

            # 瞬时CPU和内存利用率（用于绘图）
            'cpu_utilization_instant': self.cpu_utilization_history,
            'memory_utilization_instant': self.memory_utilization_history,
        })

        # 如果有成本组成部分的历史，添加到结果中
        if hasattr(self, 'cost_components_history') and self.cost_components_history:
            for component in ['cluster_risk', 'power_consumption', 'migration_cost', 'resource_contention']:
                results[component] = [history.get(component, 0.0) for history in self.cost_components_history]

        # 导出到CSV
        results.to_csv(file_path, index=False)
        print(f"Results exported to {file_path}")

        # 打印统计摘要
        print("\n=== 运行结果统计摘要 ===")
        print(f"总时间窗口数: {len(self.costs_history)}")
        print(f"总共居时长: {self.total_cohabitation_time}")
        print(f"总迁移次数: {self.total_migrations}")
        print(f"平均CPU利用率 (0-t): {np.mean(self.cpu_utilization_history):.2f}%")
        print(f"平均内存利用率 (0-t): {np.mean(self.memory_utilization_history):.2f}%")
        print(f"当前服务器总风险度 (最后一个时间窗口): {self.server_total_risk_history[-1]:.2f}")
        print(f"当前VM平均风险值 (最后一个时间窗口): {self.vm_average_risk_history[-1]:.2f}")
        print(f"当前共居恶意VM数量 (最后一个时间窗口): {self.cohabitation_malicious_vm_count_history[-1]}")
        print(f"当前开启服务器数量 (最后一个时间窗口): {self.active_servers_history[-1]}")

def main():
    """主函数"""
    # 设置虚拟机和物理机数量
    VM_COUNT = 100
    PM_COUNT = 10

    # 创建并运行MPC调度器
    scheduler = MPCScheduler(vm_count=VM_COUNT, pm_count=PM_COUNT)
    sandpiper = MPCScheduler(vm_count=VM_COUNT, pm_count=PM_COUNT)
    metis = MPCScheduler(vm_count=VM_COUNT, pm_count=PM_COUNT)

    # 训练模型
    scheduler.ensure_models_trained()

    # 运行MPC优化
    # 设置开始和结束的时间窗口
    START_WINDOW = 0
    END_WINDOW = None  # None表示运行到最后一个窗口
    PREDICTION_STEPS = 3  # 预测未来3个时间窗口

    scheduler.run_mpc_optimization(
        start_window=START_WINDOW,
        end_window=END_WINDOW,
        prediction_steps=PREDICTION_STEPS
    )

    # 绘制并保存结果
    print("\n=== 生成图表 ===")

    # 1. 原始的综合结果图
    scheduler.plot_results(save_path="mpc_results_comprehensive.png")

    # 2. 时间序列折线图（共居时长、CPU使用率、内存使用率、迁移次数）
    scheduler.plot_time_series_charts(save_path="mpc_time_series.png")

    # 3. 风险对比柱状图（VM风险值、服务器总风险度-不同物理机数量）
    scheduler.plot_risk_comparison_charts(save_path="mpc_risk_comparison.png")

    # 导出结果到CSV
    scheduler.export_results("mpc_results_detailed_new_829.csv")


if __name__ == "__main__":
    main()