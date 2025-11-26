import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set


class DataLoader:
    def __init__(self, vm_count: int = 50, pm_count: int = 39, join_ratio: float = 0.1):
        """
        初始化数据加载器
        :param vm_count: 虚拟机数量
        :param pm_count: 物理机数量
        :param join_ratio: 中途加入的虚拟机比例 (0.0-1.0)，仅影响数据可见性与参与计算的时间
        """
        self.vm_count = vm_count
        self.pm_count = pm_count
        self.time_window_size = 50  # 每个时间窗口的行数（0.1s * 50 = 5s）

        # 存储数据的变量
        self._vm_pm_mapping_raw = {}  # 原始虚拟机到物理机的映射（不随窗口过滤）
        self.pm_ids = []  # 物理机ID列表
        self.vm_instances = {}  # 虚拟机实例数据 {vm_id: pd.DataFrame}
        self.time_windows = {}  # 按时间窗口聚合的数据 {vm_id: {time_window_id: {cpu, memory, risk}}}
        # 当 vm_count > 100 时，>100 的 VM 映射到 51-100 的基础VM，避免额外训练与预测
        self.vm_alias = {}  # {vm_id(>100): base_vm_id in [51,100]}
        # VM 到达窗口（用于模拟“中途加入”）：到达窗口之前不参与任何窗口的聚合与计算
        self.join_ratio = max(0.0, min(1.0, float(join_ratio)))
        self.vm_arrival_window = {}  # {vm_id: arrival_window_id}
        self.current_window = None  # 最近一次访问的数据窗口，用于按窗口过滤活跃VM/PM

    @property
    def vm_pm_mapping(self) -> Dict[int, int]:
        """
        基于当前窗口过滤后的VM->PM映射：仅返回已在当前窗口“已到达”的VM映射。
        若当前窗口未知，则返回原始映射。
        """
        if self.current_window is None:
            return self._vm_pm_mapping_raw
        window_id = int(self.current_window)
        filtered = {}
        for vm_id, pm_id in self._vm_pm_mapping_raw.items():
            vm_windows = self.time_windows.get(vm_id, {})
            # 仅当该VM在当前窗口存在且非占位（未到达）时，才认为其在当前映射中有效
            if window_id in vm_windows and not vm_windows[window_id].get('_inactive', False):
                filtered[vm_id] = pm_id
        return filtered

    @vm_pm_mapping.setter
    def vm_pm_mapping(self, value: Dict[int, int]):
        self._vm_pm_mapping_raw = dict(value) if value is not None else {}

    def load_vm_pm_mapping(self, file_path: str = "data/container_machine_id_10.csv") -> Dict[int, int]:
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
        self._vm_pm_mapping_raw = dict(zip(mapping_data['vm_id'], mapping_data['pm_id']))
        return self._vm_pm_mapping_raw

    def load_pm_ids(self, file_path: str = "data/mac_keys/10.csv") -> List[int]:
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

        # 仅为前100台VM加载真实数据；>100的VM仅建立别名到[51,100]
        max_base_vm = min(self.vm_count, 100)
        for vm_id in range(1, max_base_vm + 1):
            file_path = os.path.join(data_dir, f"instance_{vm_id}.csv")
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

        # 为超过100的VM建立别名到[51,100]，不再单独加载数据文件
        if self.vm_count > 100:
            for vm_id in range(101, self.vm_count + 1):
                base_vm = 50 + ((vm_id - 1) % 50) + 1  # 映射到 [51,100]
                if base_vm < 51:
                    base_vm = 51
                if base_vm > 100:
                    base_vm = 100
                self.vm_alias[vm_id] = base_vm

        # 计算各基础VM的窗口数以规划到达时间
        base_vm_window_counts = {}
        for vm_id, df in self.vm_instances.items():
            base_vm_window_counts[vm_id] = len(df) // self.time_window_size

        # 规划中途加入：选择后段VM作为延迟加入，均匀分配到达窗口
        if self.join_ratio > 0.0 and self.vm_instances:
            base_vm_ids = sorted(list(self.vm_instances.keys()))
            # 候选池限定为 1-50 号VM（若不足50则取有的）
            candidate_pool = [vm for vm in base_vm_ids if vm <= 50]
            num_joiners = int(len(candidate_pool) * self.join_ratio)
            num_joiners = max(0, min(num_joiners, len(candidate_pool)))
            joiner_vm_ids = random.sample(candidate_pool, num_joiners) if num_joiners > 0 else []

            # 确定一个安全的最大窗口（使用这些VM各自窗口数的最小值，避免越界）
            if joiner_vm_ids:
                max_safe_window = min(base_vm_window_counts.get(v, 0) for v in joiner_vm_ids)
                # 到达窗口范围 [1, max_safe_window-1]，平均分布
                arrival_points = []
                total_slots = max_safe_window - 1
                if total_slots <= 1:
                    arrival_points = [1 for _ in joiner_vm_ids]
                else:
                    for i in range(len(joiner_vm_ids)):
                        # 均匀分布到达点，避免0窗口
                        pos = 1 + int((i + 1) * total_slots / (len(joiner_vm_ids) + 1))
                        arrival_points.append(max(1, min(pos, max_safe_window - 1)))

                # 设置 joiner 的到达窗口
                for vm_id, arrive_w in zip(joiner_vm_ids, arrival_points):
                    self.vm_arrival_window[vm_id] = int(arrive_w)

            # 非 joiner 默认从窗口0开始
            for vm_id in base_vm_ids:
                if vm_id not in self.vm_arrival_window:
                    self.vm_arrival_window[vm_id] = 0
        else:
            # 所有VM默认从窗口0开始
            for vm_id in self.vm_instances.keys():
                self.vm_arrival_window[vm_id] = 0

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
                # 若该VM尚未到达，则写入占位条目（不参与计算）
                arrival_w = int(self.vm_arrival_window.get(vm_id, 0))
                if window_id < arrival_w:
                    if window_id == 0:
                        print(f"VM {vm_id}: 将于窗口 {arrival_w} 加入，之前窗口不参与计算")
                    # 用占位标记，保持窗口计数一致，但在读取时会过滤掉
                    self.time_windows[vm_id][window_id] = {
                        '_inactive': True
                    }
                    continue
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
                    'risk_level': risk_level,
                    '_inactive': False
                }

                # 每50个窗口打印一次调试信息
                if window_id % 50 == 0:
                    print(f"  窗口 {window_id}: CPU={avg_cpu:.2f}, Memory={avg_memory:.2f}, Risk={risk_level:.2f}")

        # 打印所有基础VM的时间窗口统计
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
        # 记录当前窗口，用于后续按窗口过滤活跃PM/VM
        self.current_window = int(window_id)

        result = {}

        # 先填充基础VM的数据（仅包含已到达的条目）
        for vm_id, windows in self.time_windows.items():
            if window_id in windows and not windows[window_id].get('_inactive', False):
                result[vm_id] = {k: v for k, v in windows[window_id].items() if k != '_inactive'}

        # 再为别名VM填充其基础VM的数据（仅当基础VM已到达时）
        if self.vm_alias:
            for alias_vm, base_vm in self.vm_alias.items():
                base_windows = self.time_windows.get(base_vm, {})
                if window_id in base_windows and not base_windows[window_id].get('_inactive', False):
                    result[alias_vm] = {k: v for k, v in base_windows[window_id].items() if k != '_inactive'}

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
        # 如果是别名VM，转到基础VM取序列
        base_vm_id = vm_id
        if vm_id not in self.time_windows and vm_id in self.vm_alias:
            base_vm_id = self.vm_alias[vm_id]

        if base_vm_id in self.time_windows:
            for window_id in range(start_window, end_window + 1):
                if window_id in self.time_windows[base_vm_id] and not self.time_windows[base_vm_id][window_id].get(
                        '_inactive', False):
                    result.append(
                        {k: v for k, v in self.time_windows[base_vm_id][window_id].items() if k != '_inactive'})
                else:
                    # 未到达或窗口不存在，使用零值替代
                    result.append({
                        'cpu_usage': 0.0,
                        'memory_usage': 0.0,
                        'risk_level': 0.0
                    })
        return result

    def get_active_physical_machines(self, vm_pm_mapping: Dict[int, int] = None) -> Set[int]:
        """
        获取当前活跃的物理机集合
        :param vm_pm_mapping: 虚拟机到物理机的映射，如果为None则使用当前映射
        :return: 活跃物理机ID集合
        """
        if vm_pm_mapping is None:
            vm_pm_mapping = self.vm_pm_mapping

        # 若已知当前窗口，仅统计当前窗口“已到达”的VM所在物理机
        if self.current_window is not None:
            window_id = int(self.current_window)
            active_pms = set()
            for vm_id, pm_id in vm_pm_mapping.items():
                vm_windows = self.time_windows.get(vm_id, {})
                if window_id in vm_windows and not vm_windows[window_id].get('_inactive', False):
                    active_pms.add(pm_id)
            return active_pms

        # 回退：无窗口上下文，则返回映射中的所有PM
        return set(vm_pm_mapping.values())

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

        if self.vm_alias:
            sample_alias = list(self.vm_alias.items())[:5]
            print(f"  已建立别名VM数量: {len(self.vm_alias)}，示例: {sample_alias}")

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