import numpy as np
import random
from typing import Dict, List, Tuple, Set
import gc


class MPCController:
    def __init__(self, data_loader, prediction_models, params=None):
        """
        初始化MPC控制器
        :param data_loader: 数据加载器实例
        :param prediction_models: 预测模型实例
        :param params: MPC参数字典
        """
        self.data_loader = data_loader
        self.prediction_models = prediction_models

        # 设置MPC参数
        default_params = {
            'alpha': 0.4,  # 共居风险权重
            'beta': 0.3,  # 功耗权重
            'gamma': 0.2,  # 迁移成本权重
            'delta': 0.1,  # 资源争用权重
            'nu': 1000.0,  # 迁移成本基础参数
            'mu': 1,  # 迁移成本内存系数
            'w_cpu': 0.4,  # CPU争用权重
            'w_mem': 0.6,  # 内存争用权重
            'single_server_load': 10000.0,  # 单台服务器功耗参数
            'max_servers': 20,  # 最大服务器数量，修改为物理机实际数量
            'max_resource_competition': 50000.0,  # 最大资源争用值，提高限制
            'risk_threshold': 0.5,  # 风险评级阈值，超过该值判定为攻击者
            'risk_window_threshold': 100.0,  # 风险共居时间矩阵阈值
            'attacker_clear_window': 300,  # 攻击者标记清除时间窗数量
        }

        self.params = default_params
        if params:
            self.params.update(params)

        # 初始化虚拟机状态
        self.vm_is_attacker = {}  # 记录虚拟机是否为攻击者 {vm_id: is_attacker}
        self.vm_last_attack_window = {}  # 记录虚拟机最后一次被检测为攻击的时间窗口 {vm_id: window_id}

        for vm_id in self.data_loader.vm_instances.keys():
            self.vm_is_attacker[vm_id] = False
            self.vm_last_attack_window[vm_id] = -1  # -1 表示从未被检测为攻击者

        # 初始化风险共居时间矩阵 T
        self.T = np.zeros((self.data_loader.vm_count + 1, self.data_loader.vm_count + 1))

        # 存储最近一次计算的成本组成部分
        self.last_cost_components = {
            'cluster_risk': 0.0,
            'power_consumption': 0.0,
            'migration_cost': 0.0,
            'resource_contention': 0.0,
            'total_cost': 0.0
        }

        # 记录实际执行的迁移
        self.migrations_executed = []

    def update_attacker_status(self, current_window: int):
        """
        更新攻击者状态：
        - 检查所有检测到攻击行为的虚拟机，是否满足共居条件，满足则标记为攻击者。
        - 如果攻击者连续若干时间窗口未满足“攻击 + 共居”，则取消其攻击者标记。
        :param current_window: 当前时间窗口
        """
        window_data = self.data_loader.get_time_window_data(current_window)
        vms_with_attack = []

        # 第一步：找出所有在当前窗口检测到攻击行为的虚拟机
        for vm_id, data in window_data.items():
            if data['risk_level'] > self.params['risk_threshold']:
                vms_with_attack.append(vm_id)

        # 第二步：处理攻击行为虚拟机
        for vm_id in vms_with_attack:
            attacker_pm = self.data_loader.vm_pm_mapping.get(vm_id)
            if attacker_pm is None:
                continue

            cohabiting_vms = [other_vm for other_vm, pm_id in self.data_loader.vm_pm_mapping.items()
                              if pm_id == attacker_pm and other_vm != vm_id]
            has_normal_users = any(not self.vm_is_attacker.get(other_vm, False) for other_vm in cohabiting_vms)

            if has_normal_users and cohabiting_vms:
                # 满足攻击 + 共居条件
                if not self.vm_is_attacker.get(vm_id, False):
                    self.vm_is_attacker[vm_id] = True
                    print(f"虚拟机 {vm_id} 在时间窗口 {current_window} 被标记为攻击者")
                    print(f"  - 风险评级: {window_data[vm_id]['risk_level']}")
                    print(f"  - 所在物理机: {attacker_pm}")
                    print(
                        f"  - 共居的正常用户: {[vm for vm in cohabiting_vms if not self.vm_is_attacker.get(vm, False)]}")

                # 更新最近一次满足“攻击+共居”条件的时间窗口
                self.vm_last_attack_window[vm_id] = current_window
            else:
                if not self.vm_is_attacker.get(vm_id, False):
                    print(f"虚拟机 {vm_id} 在时间窗口 {current_window} 检测到攻击行为但不满足共居条件")
                    print(f"  - 风险评级: {window_data[vm_id]['risk_level']}")
                    print(f"  - 共居虚拟机数量: {len(cohabiting_vms)}")
                    print(
                        f"  - 共居正常用户数量: {len([vm for vm in cohabiting_vms if not self.vm_is_attacker.get(vm, False)])}")

        # 第三步：取消攻击者标记的逻辑更新
        for vm_id in list(self.vm_is_attacker.keys()):
            if self.vm_is_attacker.get(vm_id, False):
                last_attack_cohab_window = self.vm_last_attack_window.get(vm_id, -1)
                if last_attack_cohab_window == -1:
                    continue
                windows_since_valid = current_window - last_attack_cohab_window
                if windows_since_valid >= self.params['attacker_clear_window']:
                    self.vm_is_attacker[vm_id] = False
                    print(f"虚拟机 {vm_id} 在时间窗口 {current_window} 取消攻击者标记")
                    print(f"  - 已连续 {windows_since_valid} 个时间窗口未满足攻击+共居条件")

    def get_vm_status(self, vm_id: int, window_id: int = None) -> Dict:
        """
        获取虚拟机状态
        :param vm_id: 虚拟机ID
        :param window_id: 时间窗口ID，如果为None则返回最新状态
        :return: 虚拟机状态字典
        """
        # 获取虚拟机是否为攻击者
        is_attacker = self.vm_is_attacker.get(vm_id, False)

        # 获取虚拟机当前所在物理机
        current_pm = self.data_loader.vm_pm_mapping.get(vm_id)

        # 获取最后一次攻击检测时间窗口
        last_attack_window = self.vm_last_attack_window.get(vm_id, -1)

        # 如果指定了时间窗口，获取该时间窗口的资源使用情况
        resource_usage = None
        if window_id is not None and vm_id in self.data_loader.time_windows:
            if window_id in self.data_loader.time_windows[vm_id]:
                resource_usage = self.data_loader.time_windows[vm_id][window_id]

        return {
            'is_attacker': is_attacker,
            'current_pm': current_pm,
            'resource_usage': resource_usage,
            'last_attack_window': last_attack_window
        }

    def update_risk_cohabitation_time(self, vm_pm_mapping: Dict[int, int] = None):
        """
        更新风险共居时间矩阵
        :param vm_pm_mapping: 虚拟机到物理机的映射，如果为None则使用当前映射
        """
        if vm_pm_mapping is None:
            vm_pm_mapping = self.data_loader.vm_pm_mapping
        
        # 为每个服务器创建虚拟机列表
        pm_to_vms = {}
        for vm_id, pm_id in vm_pm_mapping.items():
            if pm_id not in pm_to_vms:
                pm_to_vms[pm_id] = []
            pm_to_vms[pm_id].append(vm_id)
        
        # 更新风险共居时间矩阵
        for pm_id, vm_list in pm_to_vms.items():
            for attacker_vm in vm_list:
                if self.vm_is_attacker.get(attacker_vm, False):
                    for normal_vm in vm_list:
                        if not self.vm_is_attacker.get(normal_vm, False) and attacker_vm != normal_vm:
                            # 增加风险共居时间
                            self.T[attacker_vm, normal_vm] += 1
    
    def reset_risk_time_for_migrated_vms(self, migrated_vms: List[int]):
        """
        重置迁移虚拟机的风险共居时间
        :param migrated_vms: 迁移的虚拟机ID列表
        """
        for vm_id in migrated_vms:
            # 重置该VM与所有其他VM的风险共居时间
            self.T[vm_id, :] = 0
            self.T[:, vm_id] = 0
    
    def calculate_server_risk(self, pm_id: int, vm_pm_mapping: Dict[int, int] = None) -> float:
        """
        计算服务器的风险度
        :param pm_id: 物理机ID
        :param vm_pm_mapping: 虚拟机到物理机的映射，如果为None则使用当前映射
        :return: 服务器风险度
        """
        if vm_pm_mapping is None:
            vm_pm_mapping = self.data_loader.vm_pm_mapping
        
        # 获取服务器上的所有虚拟机
        pm_vms = [vm_id for vm_id, pm in vm_pm_mapping.items() if pm == pm_id]
        
        # 计算服务器风险度
        risk = 0
        for vm_a in pm_vms:
            for vm_b in pm_vms:
                if vm_a != vm_b:
                    risk += self.T[vm_a, vm_b]
        
        return risk
    
    def calculate_cluster_risk(self, vm_pm_mapping: Dict[int, int] = None) -> float:
        """
        计算集群的综合共居风险
        :param vm_pm_mapping: 虚拟机到物理机的映射，如果为None则使用当前映射
        :return: 集群综合共居风险
        """
        if vm_pm_mapping is None:
            vm_pm_mapping = self.data_loader.vm_pm_mapping
        
        # 获取活跃的物理机列表
        active_pms = self.data_loader.get_active_physical_machines(vm_pm_mapping)
        
        # 计算每台服务器的风险度
        server_risks = {}
        for pm_id in active_pms:
            server_risks[pm_id] = self.calculate_server_risk(pm_id, vm_pm_mapping)
        
        # 计算集群综合共居风险 - 风险平方和
        cluster_risk = sum(risk ** 2 for risk in server_risks.values())
        
        return cluster_risk
    
    def calculate_power_consumption(self, vm_pm_mapping: Dict[int, int] = None) -> float:
        """
        计算功耗
        :param vm_pm_mapping: 虚拟机到物理机的映射，如果为None则使用当前映射
        :return: 功耗值
        """
        if vm_pm_mapping is None:
            vm_pm_mapping = self.data_loader.vm_pm_mapping
        
        # 获取活跃的物理机数量
        active_pms = self.data_loader.get_active_physical_machines(vm_pm_mapping)
        
        # 计算功耗
        power_consumption = len(active_pms) * self.params['single_server_load']
        
        return power_consumption
    
    def calculate_migration_cost(self, current_mapping: Dict[int, int], new_mapping: Dict[int, int], window_data: Dict[int, Dict]) -> float:
        """
        计算迁移成本
        :param current_mapping: 当前虚拟机到物理机的映射
        :param new_mapping: 新的虚拟机到物理机的映射
        :param window_data: 当前时间窗口的数据
        :return: 迁移成本
        """
        migration_cost = 0
        
        for vm_id, new_pm in new_mapping.items():
            if vm_id in current_mapping and current_mapping[vm_id] != new_pm:
                # 虚拟机需要迁移
                mem_usage = window_data.get(vm_id, {}).get('memory_usage', 0)
                vm_migration_cost = self.params['nu'] + self.params['mu'] * mem_usage
                migration_cost += vm_migration_cost
        
        return migration_cost
    
    def get_migrated_vms(self, current_mapping: Dict[int, int], new_mapping: Dict[int, int]) -> List[int]:
        """
        获取需要迁移的虚拟机列表
        :param current_mapping: 当前虚拟机到物理机的映射
        :param new_mapping: 新的虚拟机到物理机的映射
        :return: 需要迁移的虚拟机ID列表
        """
        migrated_vms = []
        
        for vm_id, new_pm in new_mapping.items():
            if vm_id in current_mapping and current_mapping[vm_id] != new_pm:
                migrated_vms.append(vm_id)
        
        return migrated_vms
    
    def calculate_resource_contention(self, vm_pm_mapping: Dict[int, int], window_data: Dict[int, Dict]) -> float:
        """
        计算负载不平衡度
        :param vm_pm_mapping: 虚拟机到物理机的映射
        :param window_data: 当前时间窗口的各VM cpu/mem 使用
        :return: 负载不平衡度值
        """
        active_pms = list(self.data_loader.get_active_physical_machines(vm_pm_mapping))
        num_active = len(active_pms)
        if num_active == 0:
            return 0.0
        if num_active == 1:
            return 0.0  # 只有一台服务器，不存在不平衡

        # 聚合每台服务器的CPU与内存总使用
        pm_cpu_sum = {pm: 0.0 for pm in active_pms}
        pm_mem_sum = {pm: 0.0 for pm in active_pms}

        for vm_id, pm_id in vm_pm_mapping.items():
            if pm_id not in pm_cpu_sum:
                # 如果该PM当前未被认为是活跃（极端情况），跳过
                continue
            vm_data = window_data.get(vm_id)
            if not vm_data:
                continue
            pm_cpu_sum[pm_id] += float(vm_data.get('cpu_usage', 0.0))
            pm_mem_sum[pm_id] += float(vm_data.get('memory_usage', 0.0))

        # 计算全局平均
        cpu_avg = sum(pm_cpu_sum.values()) / num_active
        mem_avg = sum(pm_mem_sum.values()) / num_active

        w_cpu = float(self.params.get('w_cpu', 1.0))
        w_mem = float(self.params.get('w_mem', 1.0))

        # 计算不平衡度
        imbalance_sum = 0.0
        for pm in active_pms:
            cpu_dev = pm_cpu_sum[pm] - cpu_avg
            mem_dev = pm_mem_sum[pm] - mem_avg
            imbalance_sum += w_cpu * (cpu_dev * cpu_dev) + w_mem * (mem_dev * mem_dev)

        load_imbalance = imbalance_sum / (num_active - 1)
        return load_imbalance
    
    def calculate_cost(self, vm_pm_mapping: Dict[int, int], current_mapping: Dict[int, int], window_data: Dict[int, Dict]) -> float:
        """
        计算总成本
        :param vm_pm_mapping: 虚拟机到物理机的映射
        :param current_mapping: 当前虚拟机到物理机的映射
        :param window_data: 当前时间窗口的数据
        :return: 总成本
        """
        # 计算各个成本
        cluster_risk = self.calculate_cluster_risk(vm_pm_mapping)
        power_consumption = self.calculate_power_consumption(vm_pm_mapping)
        migration_cost = self.calculate_migration_cost(current_mapping, vm_pm_mapping, window_data)
        resource_contention = self.calculate_resource_contention(vm_pm_mapping, window_data)
        
        # 检查约束条件
        active_pms = self.data_loader.get_active_physical_machines(vm_pm_mapping)
        if len(active_pms) > self.params['max_servers'] or resource_contention > self.params['max_resource_competition']:
            print(f"约束条件违反: 活跃物理机数量={len(active_pms)}/{self.params['max_servers']}, 资源争用={resource_contention:.2f}/{self.params['max_resource_competition']}")
            return float('inf')  # 违反约束条件，返回无穷大成本
        
        # 计算加权总成本
        total_cost = (
            self.params['alpha'] * cluster_risk +
            self.params['beta'] * power_consumption +
            self.params['gamma'] * migration_cost +
            self.params['delta'] * resource_contention
        )
        
        # 保存各项成本到属性中，便于外部访问
        self.last_cost_components = {
            'cluster_risk': cluster_risk,
            'power_consumption': power_consumption,
            'migration_cost': migration_cost,
            'resource_contention': resource_contention,
            'total_cost': total_cost
        }
        
        return total_cost
    
    def generate_new_mapping(self, vm_pm_mapping: Dict[int, int], window_data: Dict[int, Dict], method: str = 'random', n_attempts: int = 10) -> Dict[int, int]:
        """
        生成新的虚拟机到物理机的映射
        :param vm_pm_mapping: 当前虚拟机到物理机的映射
        :param window_data: 当前时间窗口的数据
        :param method: 生成方法，可选 'random'（随机生成）或 'heuristic'（启发式生成）
        :param n_attempts: 尝试次数
        :return: 新的虚拟机到物理机的映射
        """
        # 检查是否有风险超过阈值的情况
        cluster_risk = self.calculate_cluster_risk(vm_pm_mapping)
        if cluster_risk < self.params['risk_window_threshold']:
            return vm_pm_mapping.copy()  # 如果风险较低，则不改变映射
        
        best_mapping = vm_pm_mapping.copy()
        best_cost = self.calculate_cost(vm_pm_mapping, vm_pm_mapping, window_data)
        
        for _ in range(n_attempts):
            if method == 'random':
                # 随机选择一些虚拟机进行迁移
                new_mapping = vm_pm_mapping.copy()
                
                # 随机选择1-3个虚拟机进行迁移
                n_vms_to_migrate = random.randint(1, min(3, len(vm_pm_mapping)))
                vms_to_migrate = random.sample(list(vm_pm_mapping.keys()), n_vms_to_migrate)
                
                # 随机选择目标物理机
                active_pms = list(self.data_loader.get_active_physical_machines(vm_pm_mapping))
                all_pms = self.data_loader.pm_ids
                
                for vm_id in vms_to_migrate:
                    current_pm = vm_pm_mapping[vm_id]
                    possible_pms = [pm for pm in all_pms if pm != current_pm]
                    if possible_pms:
                        new_pm = random.choice(possible_pms)
                        new_mapping[vm_id] = new_pm
                    # # 如果有多台活跃的物理机，随机选择一台不同的物理机
                    # if len(active_pms) > 1:
                    #     current_pm = vm_pm_mapping[vm_id]
                    #     possible_pms = [pm for pm in active_pms if pm != current_pm]
                    #     if possible_pms:
                    #         new_pm = random.choice(possible_pms)
                    #         new_mapping[vm_id] = new_pm
                    # # 否则随机选择一台物理机（可能是新的物理机）
                    # else:
                    #     new_pm = random.choice(all_pms)
                    #     new_mapping[vm_id] = new_pm
            
            elif method == 'heuristic':
                # 启发式方法：将高风险虚拟机分散到不同物理机
                new_mapping = vm_pm_mapping.copy()
                
                # 获取每台物理机的风险值
                pm_risks = {}
                for pm_id in self.data_loader.get_active_physical_machines(vm_pm_mapping):
                    pm_risks[pm_id] = self.calculate_server_risk(pm_id, vm_pm_mapping)
                
                # 找出风险最高的物理机
                if pm_risks:
                    highest_risk_pm = max(pm_risks.items(), key=lambda x: x[1])[0]
                    
                    # 获取这台物理机上的所有虚拟机
                    pm_vms = [vm_id for vm_id, pm in vm_pm_mapping.items() if pm == highest_risk_pm]
                    
                    # 找出这台物理机上攻击者和被攻击者的虚拟机
                    attackers = [vm_id for vm_id in pm_vms if self.vm_is_attacker.get(vm_id, False)]
                    victims = [vm_id for vm_id in pm_vms if not self.vm_is_attacker.get(vm_id, False)]
                    
                    # 如果有攻击者，尝试迁移它们
                    vms_to_migrate = attackers if attackers else (victims if victims else [])
                    
                    if vms_to_migrate:
                        # 随机选择1-3个虚拟机进行迁移
                        n_vms_to_migrate = min(len(vms_to_migrate), 5)
                        vms_to_migrate = random.sample(vms_to_migrate, n_vms_to_migrate)
                        
                        # 获取所有可用的物理机
                        all_pms = self.data_loader.pm_ids
                        
                        # 寻找风险最低的物理机作为目标
                        lowest_risk_pms = sorted(pm_risks.items(), key=lambda x: x[1])
                        target_pms = [pm for pm, _ in lowest_risk_pms if pm != highest_risk_pm]
                        
                        # 如果没有其他活跃的物理机，随机选择一台未使用的物理机
                        if not target_pms:
                            inactive_pms = [pm for pm in all_pms if pm not in pm_risks]
                            if inactive_pms:
                                target_pms = [random.choice(inactive_pms)]
                        
                        # 迁移虚拟机
                        if target_pms:
                            for i, vm_id in enumerate(vms_to_migrate):
                                target_pm = target_pms[i % len(target_pms)]
                                new_mapping[vm_id] = target_pm
            
            # 计算新映射的成本
            new_cost = self.calculate_cost(new_mapping, vm_pm_mapping, window_data)
            
            # 如果新映射的成本更低，则更新最佳映射
            if new_cost < best_cost:
                best_mapping = new_mapping.copy()
                best_cost = new_cost
        
        return best_mapping
    
    def optimize_vm_placement(self, current_window: int) -> Dict[int, int]:
        """
        优化虚拟机放置
        :param current_window: 当前时间窗口
        :return: 优化后的虚拟机到物理机的映射
        """
        # 获取当前窗口的数据
        window_data = self.data_loader.get_time_window_data(current_window)
        
        # 更新攻击者状态
        self.update_attacker_status(current_window)
        
        # 当前虚拟机到物理机的映射
        current_mapping = self.data_loader.vm_pm_mapping.copy()
        
        # 计算当前成本
        current_cost = self.calculate_cost(current_mapping, current_mapping, window_data)
        
        # 输出详细的成本信息
        if current_cost != float('inf'):
            print(f"时间窗口 {current_window} 当前成本详情:")
            print(f"  - 综合共居风险: {self.last_cost_components['cluster_risk']:.2f} × {self.params['alpha']} = {self.params['alpha'] * self.last_cost_components['cluster_risk']:.2f}")
            print(f"  - 功耗: {self.last_cost_components['power_consumption']:.2f} × {self.params['beta']} = {self.params['beta'] * self.last_cost_components['power_consumption']:.2f}")
            print(f"  - 迁移成本: {self.last_cost_components['migration_cost']:.2f} × {self.params['gamma']} = {self.params['gamma'] * self.last_cost_components['migration_cost']:.2f}")
            print(f"  - 资源争用: {self.last_cost_components['resource_contention']:.2f} × {self.params['delta']} = {self.params['delta'] * self.last_cost_components['resource_contention']:.2f}")
            print(f"  - 总成本: {current_cost:.2f}")
        else:
            print(f"时间窗口 {current_window} 的当前成本: {current_cost}")
        
        # 尝试优化放置
        methods = ['random', 'heuristic']
        best_mapping = current_mapping
        best_cost = current_cost
        
        for method in methods:
            # 生成新的映射
            new_mapping = self.generate_new_mapping(current_mapping, window_data, method=method, n_attempts=20)
            
            # 计算新的成本
            new_cost = self.calculate_cost(new_mapping, current_mapping, window_data)
            
            # 如果新的成本更低，更新最佳映射
            if new_cost < best_cost:
                best_mapping = new_mapping
                best_cost = new_cost
        
        # 保存优化后的成本，以便外部使用
        self.last_optimization_cost = best_cost
        
        # 如果有更优的映射，应用它
        if best_mapping != current_mapping and best_cost < current_cost:
            # 获取需要迁移的虚拟机
            migrated_vms = self.get_migrated_vms(current_mapping, best_mapping)
            
            if migrated_vms:
                print(f"时间窗口 {current_window} 执行迁移: {migrated_vms}")
                
                # 输出详细的迁移信息
                print("迁移详情:")
                for vm_id in migrated_vms:
                    source_pm = current_mapping[vm_id]
                    target_pm = best_mapping[vm_id]
                    print(f"  - 虚拟机 {vm_id} 从物理机 {source_pm} 迁移到物理机 {target_pm}")
                
                print(f"迁移前成本: {current_cost:.2f}, 迁移后成本: {best_cost:.2f}")
                
                if best_cost != float('inf'):
                    print(f"优化后成本详情:")
                    print(f"  - 综合共居风险: {self.last_cost_components['cluster_risk']:.2f} × {self.params['alpha']} = {self.params['alpha'] * self.last_cost_components['cluster_risk']:.2f}")
                    print(f"  - 功耗: {self.last_cost_components['power_consumption']:.2f} × {self.params['beta']} = {self.params['beta'] * self.last_cost_components['power_consumption']:.2f}")
                    print(f"  - 迁移成本: {self.last_cost_components['migration_cost']:.2f} × {self.params['gamma']} = {self.params['gamma'] * self.last_cost_components['migration_cost']:.2f}")
                    print(f"  - 资源争用: {self.last_cost_components['resource_contention']:.2f} × {self.params['delta']} = {self.params['delta'] * self.last_cost_components['resource_contention']:.2f}")
                
                # 重置迁移虚拟机的风险共居时间
                self.reset_risk_time_for_migrated_vms(migrated_vms)
                
                # 更新虚拟机到物理机的映射
                self.data_loader.vm_pm_mapping = best_mapping
                
                # 记录实际执行的迁移
                self.migrations_executed = migrated_vms
            else:
                self.migrations_executed = []
                print(f"时间窗口 {current_window} 无需迁移")
        else:
            self.migrations_executed = []
            print(f"时间窗口 {current_window} 无需迁移")
        
        # 更新风险共居时间矩阵
        self.update_risk_cohabitation_time()
        
        # 清理内存
        gc.collect()
        
        return self.data_loader.vm_pm_mapping 