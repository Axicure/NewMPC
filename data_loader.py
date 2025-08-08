import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set


class DataLoader:
    def __init__(self, vm_count: int = 50, pm_count: int = 39):
        """
        初始化数据加载器
        :param vm_count: 虚拟机数量
        :param pm_count: 物理机数量
        """
        self.vm_count = vm_count
        self.pm_count = pm_count
        self.time_window_size = 50  # 每个时间窗口的行数（0.1s * 50 = 5s）

        # 存储数据的变量
        self.vm_pm_mapping = {}  # 虚拟机到物理机的映射
        self.pm_ids = []  # 物理机ID列表
        self.vm_instances = {}  # 虚拟机实例数据 {vm_id: pd.DataFrame}
        self.time_windows = {}  # 按时间窗口聚合的数据 {vm_id: {time_window_id: {cpu, memory, risk}}}

    def load_vm_pm_mapping(self, file_path: str = "data/container_machine_id_20.csv") -> Dict[int, int]:
        """
        加载虚拟机和物理机的映射关系
        :param file_path: 文件路径
        :return: 虚拟机到物理机的映射字典
        """
        mapping_data = pd.read_csv(file_path, header=None)
        mapping_data.columns = ['vm_id', 'pm_id']

        # 限制为指定数量的虚拟机
        mapping_data = mapping_data[mapping_data['vm_id'] <= self.vm_count]

        # 转换为字典
        self.vm_pm_mapping = dict(zip(mapping_data['vm_id'], mapping_data['pm_id']))
        return self.vm_pm_mapping

    def load_pm_ids(self, file_path: str = "data/mac_keys/20.csv") -> List[int]:
        """
        加载物理机ID列表
        :param file_path: 文件路径
        :return: 物理机ID列表
        """
        pm_data = pd.read_csv(file_path)

        # 限制为指定数量的物理机
        self.pm_ids = pm_data['macid'].values.tolist()[:self.pm_count]
        return self.pm_ids

    def load_vm_instance_data(self, data_dir: str = "data/target") -> Dict[int, pd.DataFrame]:
        """
        加载虚拟机实例数据
        :param data_dir: 数据目录
        :return: 虚拟机实例数据字典
        """
        self.vm_instances = {}

        for vm_id in range(1, self.vm_count + 1):
            if vm_id < 100:
                file_path = os.path.join(data_dir, f"instance_{vm_id}.csv")
            else:
                file_path = os.path.join(data_dir, f"instance_{vm_id % 50 + 50}.csv")
            if os.path.exists(file_path):
                # 读取数据文件 - 没有表头的CSV文件
                df = pd.read_csv(file_path, header=None)
                df.columns = ['cpu_usage', 'memory_usage', 'risk_level']

                # 数据完整性检查
                print(f"VM {vm_id}: 读取 {len(df)} 行数据")
                print(f"  CPU使用率范围: {df['cpu_usage'].min():.2f} - {df['cpu_usage'].max():.2f}")
                print(f"  内存使用率范围: {df['memory_usage'].min():.2f} - {df['memory_usage'].max():.2f}")
                print(f"  风险级别范围: {df['risk_level'].min():.2f} - {df['risk_level'].max():.2f}")

                # 检查是否有NaN值
                if df.isnull().any().any():
                    print(f"  警告: VM {vm_id} 数据中存在NaN值")
                    df = df.fillna(0)  # 用0填充NaN值

                self.vm_instances[vm_id] = df
            else:
                print(f"警告: 找不到虚拟机 {vm_id} 的数据文件 {file_path}")

        return self.vm_instances

    def aggregate_data_by_time_window(self) -> Dict[int, Dict[int, Dict]]:
        """
        按时间窗口聚合数据
        :return: 按时间窗口聚合的数据字典
        """
        self.time_windows = {}

        for vm_id, df in self.vm_instances.items():
            self.time_windows[vm_id] = {}

            # 计算时间窗口数量
            num_windows = len(df) // self.time_window_size

            print(f"VM {vm_id}: 原始数据 {len(df)} 行, 生成 {num_windows} 个时间窗口")

            for window_id in range(num_windows):
                start_idx = window_id * self.time_window_size
                end_idx = start_idx + self.time_window_size

                # 获取当前窗口的数据
                window_data = df.iloc[start_idx:end_idx]

                # 检查窗口数据是否完整
                if len(window_data) != self.time_window_size:
                    print(f"  警告: VM {vm_id} 窗口 {window_id} 数据不完整: {len(window_data)}/{self.time_window_size}")

                # 计算平均值
                avg_cpu = window_data['cpu_usage'].mean()
                avg_memory = window_data['memory_usage'].mean()
                # 风险级别在每个时间窗口内是相同的，取第一个值即可
                risk_level = window_data['risk_level'].iloc[0] if len(window_data) > 0 else 0.0

                # 检查计算结果
                if np.isnan(avg_cpu) or np.isnan(avg_memory):
                    print(f"  警告: VM {vm_id} 窗口 {window_id} 计算出现NaN值")
                    avg_cpu = 0.0 if np.isnan(avg_cpu) else avg_cpu
                    avg_memory = 0.0 if np.isnan(avg_memory) else avg_memory

                # 存储聚合数据
                self.time_windows[vm_id][window_id] = {
                    'cpu_usage': avg_cpu,
                    'memory_usage': avg_memory,
                    'risk_level': risk_level
                }

                # 每50个窗口打印一次调试信息
                if window_id % 50 == 0:
                    print(f"  窗口 {window_id}: CPU={avg_cpu:.2f}, Memory={avg_memory:.2f}, Risk={risk_level:.2f}")

        # 打印所有VM的时间窗口统计
        window_counts = [len(windows) for windows in self.time_windows.values()]
        print(f"\n时间窗口统计:")
        print(f"  最少窗口数: {min(window_counts) if window_counts else 0}")
        print(f"  最多窗口数: {max(window_counts) if window_counts else 0}")
        print(f"  平均窗口数: {np.mean(window_counts) if window_counts else 0:.2f}")

        return self.time_windows

    def get_time_window_data(self, window_id: int) -> Dict[int, Dict]:
        """
        获取指定时间窗口的所有虚拟机数据
        :param window_id: 时间窗口ID
        :return: 该时间窗口的所有虚拟机数据
        """
        result = {}
        missing_vms = []

        for vm_id, windows in self.time_windows.items():
            if window_id in windows:
                result[vm_id] = windows[window_id]
            else:
                missing_vms.append(vm_id)

        # 如果有VM缺少该时间窗口的数据，打印警告
        if missing_vms and window_id % 20 == 0:  # 每20个窗口打印一次
            print(f"警告: 时间窗口 {window_id} 缺少 {len(missing_vms)} 个VM的数据: {missing_vms[:5]}...")

        return result

    def get_vm_data_sequence(self, vm_id: int, start_window: int, end_window: int) -> List[Dict]:
        """
        获取指定虚拟机在时间窗口序列内的数据
        :param vm_id: 虚拟机ID
        :param start_window: 起始窗口ID
        :param end_window: 结束窗口ID
        :return: 数据序列
        """
        result = []
        if vm_id in self.time_windows:
            for window_id in range(start_window, end_window + 1):
                if window_id in self.time_windows[vm_id]:
                    result.append(self.time_windows[vm_id][window_id])
                else:
                    # 如果窗口不存在，使用零值替代
                    result.append({
                        'cpu_usage': 0.0,
                        'memory_usage': 0.0,
                        'risk_level': 0.0
                    })
                    if window_id % 50 == 0:  # 每50个窗口打印一次
                        print(f"警告: VM {vm_id} 缺少窗口 {window_id} 的数据，使用零值替代")
        return result

    def get_active_physical_machines(self, vm_pm_mapping: Dict[int, int] = None) -> Set[int]:
        """
        获取当前活跃的物理机集合
        :param vm_pm_mapping: 虚拟机到物理机的映射，如果为None则使用当前映射
        :return: 活跃物理机ID集合
        """
        if vm_pm_mapping is None:
            vm_pm_mapping = self.vm_pm_mapping

        # 获取所有有虚拟机运行的物理机
        active_pms = set(vm_pm_mapping.values())
        return active_pms

    def load_all_data(self):
        """
        加载所有数据
        """
        print("开始加载数据...")
        self.load_vm_pm_mapping()
        self.load_pm_ids()
        self.load_vm_instance_data()
        self.aggregate_data_by_time_window()

        print(f"\n数据加载完成:")
        print(f"  已加载 {len(self.vm_instances)} 个虚拟机的数据")
        print(f"  已加载 {len(self.pm_ids)} 个物理机的ID")

        if self.time_windows:
            window_counts = [len(windows) for windows in self.time_windows.values()]
            print(f"  时间窗口总数: 最少={min(window_counts)}, 最多={max(window_counts)}")

        # 数据完整性最终检查
        self.check_data_integrity()

    def check_data_integrity(self):
        """
        检查数据完整性
        """
        print("\n=== 数据完整性检查 ===")

        # 检查每个VM的时间窗口数量
        window_counts = {}
        for vm_id, windows in self.time_windows.items():
            window_counts[vm_id] = len(windows)

        min_windows = min(window_counts.values()) if window_counts else 0
        max_windows = max(window_counts.values()) if window_counts else 0

        print(f"时间窗口数量: 最少={min_windows}, 最多={max_windows}")

        # 找出时间窗口数量不一致的VM
        inconsistent_vms = [vm_id for vm_id, count in window_counts.items() if count != max_windows]
        if inconsistent_vms:
            print(f"时间窗口数量不一致的VM: {inconsistent_vms[:10]}...")  # 只显示前10个

        # 检查每个时间窗口的VM覆盖率
        coverage_stats = []
        for window_id in range(min_windows):
            window_data = self.get_time_window_data(window_id)
            coverage = len(window_data) / len(self.vm_instances) * 100
            coverage_stats.append(coverage)

        if coverage_stats:
            avg_coverage = np.mean(coverage_stats)
            min_coverage = min(coverage_stats)
            print(f"时间窗口VM覆盖率: 平均={avg_coverage:.1f}%, 最低={min_coverage:.1f}%")

            # 找出覆盖率低于95%的时间窗口
            low_coverage_windows = [i for i, cov in enumerate(coverage_stats) if cov < 95]
            if low_coverage_windows:
                print(f"覆盖率低于95%的时间窗口数量: {len(low_coverage_windows)}")
                print(f"示例时间窗口: {low_coverage_windows[:5]}")

        print("=== 数据完整性检查完成 ===\n")