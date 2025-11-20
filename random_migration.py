from pathlib import Path
from typing import List, Dict, Optional
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect 
from data_loader import DataLoader
from prediction_models import PredictionModels
from mpc_controller import MPCController

def generate_random_schedule(
    time_steps: int,
    vm_ids: List[int],
    pm_ids: List[int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成“随机间隔 1~5 个窗口触发；每次迁移 1~3 个 VM”的迁移计划
    输出列：time, vm_id, src_pm, dst_pm, action
    """
    random.seed(seed)
    np.random.seed(seed)

    placement = {int(vm): int(random.choice(pm_ids)) for vm in vm_ids}
    t = 0
    events = []

    while t < time_steps:
        interval = random.randint(1, 5)
        t += interval
        if t >= time_steps:
            break
        # 与MPC方案一致：单次迁移数量在 [1% * VM数, 5% * VM数]
        min_limit = max(1, int(len(vm_ids) * 0.01))
        max_limit = max(1, int(len(vm_ids) * 0.05))
        k = random.randint(min_limit, min(max_limit, len(vm_ids)))
        chosen_vms = random.sample(vm_ids, k)

        for vm in chosen_vms:
            vm = int(vm)
            src = placement[vm]
            if len(pm_ids) > 1:
                dst_candidates = [p for p in pm_ids if int(p) != int(src)]
                dst = int(random.choice(dst_candidates))
            else:
                dst = src
            events.append(
                {
                    "time": int(t),
                    "vm_id": int(vm),
                    "src_pm": int(src),
                    "dst_pm": int(dst),
                    "action": "migrate",
                }
            )
            placement[vm] = dst

    df = pd.DataFrame(events).sort_values(["time", "vm_id"]).reset_index(drop=True)
    return df


def _get_attacker_mask(controller, current_window: int) -> Dict[int, bool]:
    """
    更新并读取攻击者状态，返回 {vm_id: is_attacker}
    """
    try:
        controller.update_attacker_status(current_window=current_window)
    except Exception:
        pass

    for src in [
        getattr(controller, "vm_is_attacker", None),
        getattr(controller, "attackers", None),
        getattr(getattr(controller, "data_loader", None), "vm_is_attacker", None),
    ]:
        if isinstance(src, dict) and src:
            return {int(k): bool(v) for k, v in src.items()}
    return {}


def _get_window_metrics_instant(
    controller,
    t: int,
    mapping: Dict[int, int],
    pm_ids: Optional[List[int]] = None,
    normalize_by_total_pms: bool = True,
) -> Dict[str, float]:
    """
    计算“瞬时”CPU/Memory 与活跃主机数：
    - 对每个 VM 调 controller.get_vm_status(vm, t) 取 cpu_usage / memory_usage
    - 聚合到 PM
    - 若 normalize_by_total_pms=True：按“全体 PM 数”做归一平均（体现相对起点，初期更低）
      否则对活跃 PM 求均值
    """
    pm_cpu: Dict[int, float] = {}
    pm_mem: Dict[int, float] = {}

    # 使用与 MPC 一致的数据来源：data_loader.get_time_window_data(t)
    window_data = {}
    try:
        window_data = controller.data_loader.get_time_window_data(t)
    except Exception:
        window_data = {}

    for vm, pm in mapping.items():
        data = window_data.get(int(vm), {})
        cpu = float(data.get("cpu_usage", 0.0))
        mem = float(data.get("memory_usage", 0.0))
        pm_cpu[pm] = pm_cpu.get(pm, 0.0) + cpu
        pm_mem[pm] = pm_mem.get(pm, 0.0) + mem

    active_servers = len({pm for pm in mapping.values() if pm in pm_cpu or pm in pm_mem})
    total_servers = len(pm_ids) if pm_ids else active_servers

    if normalize_by_total_pms and total_servers > 0:
        cpu_inst = float(sum(pm_cpu.values()) / total_servers) if pm_cpu else 0.0
        mem_inst = float(sum(pm_mem.values()) / total_servers) if pm_mem else 0.0
    else:
        if active_servers > 0:
            cpu_inst = float(np.mean(list(pm_cpu.values()))) if pm_cpu else 0.0
            mem_inst = float(np.mean(list(pm_mem.values()))) if pm_mem else 0.0
        else:
            cpu_inst = 0.0
            mem_inst = 0.0

    return {
        "cpu_utilization_instant": cpu_inst,
        "memory_utilization_instant": mem_inst,
        "active_servers_count": int(active_servers),
    }


def compute_random_results_with_controller(
    controller,
    schedule: pd.DataFrame,
    time_steps: int,
    vm_ids: List[int],
    pm_ids: List[int],
    out_dir: str = ".",
    csv_name: str = "random_results_detailed.csv",
    consolidate_initial: bool = True,
    initial_active_pm_ratio: float = 0.2,
    initial_active_pm_count: Optional[int] = None,
    normalize_by_total_pms: bool = False,
) -> pd.DataFrame:
    """
    逐窗口执行随机迁移计划，并复用 MPC 口径计算指标。
    - consolidate_initial: t=0 将 VM 集中到少量 PM（初期 CPU/内存更低）
    - initial_active_pm_ratio / count: 初期活跃 PM 数的比例/数量
    - normalize_by_total_pms: 瞬时 CPU/内存按总 PM 数平均，体现“相对起点”
    生成列与 mpc_results_detailed.csv 完全一致。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 初始映射：按 data_loader 的初始分布（仅包含已到达的VM），并确保所有 vm_ids 都被映射
    mapping = {}
    if hasattr(controller, "data_loader") and hasattr(controller.data_loader, "vm_pm_mapping"):
        mapping.update(dict(controller.data_loader.vm_pm_mapping))
    # 对未映射的 VM 做回退均匀分配（保证所有传入 vm_ids 都有映射）
    if vm_ids and pm_ids:
        for vm in vm_ids:
            if int(vm) not in mapping:
                mapping[int(vm)] = int(pm_ids[(int(vm) - 1) % len(pm_ids)])

    # 初始不再强行集中到少量PM，保持与 MPC 一致（由数据与映射决定活跃物理机）
    # 如需保留该行为，可将 consolidate_initial 设为 True 并按需处理

    schedule = schedule.copy()
    if "src_host" in schedule.columns and "src_pm" not in schedule.columns:
        schedule = schedule.rename(columns={"src_host": "src_pm"})
    if "dst_host" in schedule.columns and "dst_pm" not in schedule.columns:
        schedule = schedule.rename(columns={"dst_host": "dst_pm"})

    rows = []
    cumulative_migs = 0
    cohab_time_cum = 0
    cpu_avg = 0.0
    mem_avg = 0.0

    # 迁移成本所需窗口数据
    def _get_window_data(loader, tt):
        for name in ("get_time_window_data", "get_window_data"):
            fn = getattr(loader, name, None)
            if callable(fn):
                try:
                    return fn(tt)
                except Exception:
                    pass
        return {}

    # 相对起点（用于绘图，CSV 仍保留原始瞬时值）
    cpu_base = None
    mem_base = None
    cpu_rel_inst_mono = 0.0
    mem_rel_inst_mono = 0.0
    cpu_rel_avg = 0.0
    mem_rel_avg = 0.0

    for t in range(int(time_steps)):
        # 1) 攻击者
        attacker_mask = _get_attacker_mask(controller, t)
        attacker_count = int(sum(1 for v in attacker_mask.values() if v))

        # 2) 应用本窗口迁移
        win = schedule[schedule["time"] == t]
        new_mapping = dict(mapping)
        migrated_vms: List[int] = []
        for _, r in win.iterrows():
            vm = int(r["vm_id"])
            dst = int(r["dst_pm"])
            if new_mapping.get(vm) != dst:
                new_mapping[vm] = dst
                migrated_vms.append(vm)

        # 3) 迁移成本
        try:
            migration_cost = controller.calculate_migration_cost(mapping, new_mapping, _get_window_data(controller.data_loader, t))
        except Exception:
            migration_cost = 0.0

        # 4) 共居时长（同 PM 上“攻击者×正常”）
        try:
            pm_to_vms: Dict[int, List[int]] = {}
            for vm_id, pm_id in new_mapping.items():
                pm_to_vms.setdefault(pm_id, []).append(vm_id)
            cohab_malicious_count = 0
            inc = 0
            for pm_id, vms in pm_to_vms.items():
                attackers = [vid for vid in vms if attacker_mask.get(vid, False)]
                normals = [vid for vid in vms if not attacker_mask.get(vid, False)]
                if attackers and normals:
                    cohab_malicious_count += len(attackers)
                    inc += len(attackers) * len(normals)
        except Exception:
            cohab_malicious_count = 0
            inc = 0
        cohab_time_cum += inc

        # 同步控制器内部（若方法存在）
        try:
            controller.reset_risk_time_for_migrated_vms(migrated_vms)
            controller.update_risk_cohabitation_time(vm_pm_mapping=new_mapping)
        except Exception:
            pass

        # 5) 风险、能耗、资源争用
        try:
            cluster_risk = controller.calculate_cluster_risk(vm_pm_mapping=new_mapping)
        except Exception:
            cluster_risk = 0.0
        try:
            power_consumption = controller.calculate_power_consumption(vm_pm_mapping=new_mapping)
        except Exception:
            power_consumption = 0.0
        try:
            resource_contention = controller.calculate_resource_contention(
                vm_pm_mapping=new_mapping,
                window_data=_get_window_data(controller.data_loader, t),
            )
        except Exception:
            resource_contention = 0.0
        try:
            server_total_risk = sum(controller.calculate_server_risk(pm, vm_pm_mapping=new_mapping) for pm in pm_ids)
        except Exception:
            server_total_risk = cluster_risk
        vm_average_risk = float(cluster_risk) / max(1, len(vm_ids))

        # 6) CPU/Memory（瞬时 + 递推平均；并计算“相对起点”的单调序列用于绘图）
        wm = _get_window_metrics_instant(
            controller, t, new_mapping, pm_ids=pm_ids, normalize_by_total_pms=normalize_by_total_pms
        )
        cpu_inst = float(wm["cpu_utilization_instant"])
        mem_inst = float(wm["memory_utilization_instant"])
        active_servers_count = wm["active_servers_count"]

        # 原始瞬时的累计平均（CSV 用）
        cpu_avg = (cpu_avg * t + cpu_inst) / (t + 1)
        mem_avg = (mem_avg * t + mem_inst) / (t + 1)

        # 相对起点（绘图用）：减首帧并裁零；做 cumulative max 保证不降
        if cpu_base is None:
            cpu_base = cpu_inst
        if mem_base is None:
            mem_base = mem_inst
        cpu_rel = max(0.0, cpu_inst - cpu_base)
        mem_rel = max(0.0, mem_inst - mem_base)
        cpu_rel_inst_mono = max(cpu_rel_inst_mono, cpu_rel)
        mem_rel_inst_mono = max(mem_rel_inst_mono, mem_rel)
        cpu_rel_avg = (cpu_rel_avg * t + cpu_rel_inst_mono) / (t + 1)
        mem_rel_avg = (mem_rel_avg * t + mem_rel_inst_mono) / (t + 1)

        # 7) 总成本（与MPC权重一致）
        alpha = float(getattr(controller, 'params', {}).get('alpha', 0.4))
        beta = float(getattr(controller, 'params', {}).get('beta', 0.3))
        gamma = float(getattr(controller, 'params', {}).get('gamma', 0.1))
        delta = float(getattr(controller, 'params', {}).get('delta', 0.2))
        cost = (
            alpha * float(cluster_risk) +
            beta * float(power_consumption) +
            gamma * float(migration_cost) +
            delta * float(resource_contention)
        )

        cumulative_migs += len(migrated_vms)

        rows.append(
            dict(
                window_id=int(t),
                cost=float(cost),
                migrations=int(len(migrated_vms)),
                attacker_count=int(attacker_count),
                cohabitation_time_cumulative=int(cohab_time_cum),
                cohabitation_malicious_vm_count=int(cohab_malicious_count),
                server_total_risk=float(server_total_risk),
                vm_average_risk=float(vm_average_risk),
                cpu_utilization_avg=float(cpu_avg),
                memory_utilization_avg=float(mem_avg),
                migrations_cumulative=int(cumulative_migs),
                active_servers_count=int(active_servers_count),
                cpu_utilization_instant=float(cpu_inst),
                memory_utilization_instant=float(mem_inst),
                cluster_risk=float(cluster_risk),
                power_consumption=float(power_consumption),
                migration_cost=float(migration_cost),
                resource_contention=float(resource_contention),
            )
        )

        # 应用新映射
        mapping = new_mapping

    df_res = pd.DataFrame(rows)
    df_res.to_csv(out / csv_name, index=False)
    print(f"[random] results saved: {out / csv_name} (rows={len(df_res)})")

    # 将“相对起点”的单调序列临时挂到结果（仅供绘图使用，不写入 CSV）
    df_res["_cpu_rel_inst_mono"] = _cumulative_max_relative(df_res["cpu_utilization_instant"].values)
    df_res["_mem_rel_inst_mono"] = _cumulative_max_relative(df_res["memory_utilization_instant"].values)
    df_res["_cpu_rel_avg"] = _running_mean(df_res["_cpu_rel_inst_mono"].values)
    df_res["_mem_rel_avg"] = _running_mean(df_res["_mem_rel_inst_mono"].values)

    return df_res


def _cumulative_max_relative(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    base = float(arr[0])
    rel = np.maximum(0.0, arr - base)
    return np.maximum.accumulate(rel)


def _running_mean(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = np.zeros_like(arr, dtype=float)
    s = 0.0
    for i, v in enumerate(arr):
        s += float(v)
        out[i] = s / (i + 1)
    return out


def plot_random_figures(df: pd.DataFrame, out_dir: str = "."):
    """
    绘制三张与 MPC 对应的图片：
    1) random_results_comprehensive.png（Cost，Components，Migrations，Attackers，Cumulative Migrations）
    2) random_time_series.png（Cohabitation Time, CPU Utilization, Memory Utilization, Cumulative Migrations）
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 图1：综合
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    axs[0].plot(df["window_id"], df["cost"], color="blue", label="Total Cost")
    axs[0].set_ylabel("Cost")
    axs[0].set_title("Random Optimization Results")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(df["window_id"], df["cluster_risk"], "r-", label="Cohabitation Risk")
    axs[1].plot(df["window_id"], df["power_consumption"], "g-", label="Power Consumption")
    axs[1].plot(df["window_id"], df["migration_cost"], "y-", label="Migration Cost")
    axs[1].plot(df["window_id"], df["resource_contention"], "m-", label="Resource Contention")
    axs[1].set_ylabel("Cost Components")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].bar(df["window_id"], df["migrations"], color="seagreen", label="Migration Count")
    axs[2].set_ylabel("Migrations")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(df["window_id"], df["attacker_count"], "r-", label="Attackers Count")
    axs[3].set_ylabel("Attackers")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    # 与 MPC 一致：使用当期 migrations 的累计和绘制
    cum_migs = np.cumsum(df["migrations"].values if "migrations" in df.columns else np.zeros(len(df)))
    axs[4].plot(df["window_id"], cum_migs, "k-", label="Cumulative Migrations")
    axs[4].set_xlabel("Time Window")
    axs[4].set_ylabel("Cumulative Migrations")
    axs[4].legend()
    axs[4].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "random_results_comprehensive.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 图2：时间序列（与 MPC 一致，使用瞬时 CPU/内存）
    fig2, ax2 = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    ax2[0, 0].plot(df["window_id"], df["cohabitation_time_cumulative"], "r-")
    ax2[0, 0].set_title("Cohabitation Time (Cumulative)")
    ax2[0, 0].set_ylabel("Cohabitation Time")
    ax2[0, 0].grid(True, alpha=0.3)

    # 与 MPC 相同：直接使用瞬时 CPU 利用率
    ax2[0, 1].plot(df["window_id"], df["cpu_utilization_instant"], "b-")
    ax2[0, 1].set_title("CPU Utilization")
    ax2[0, 1].set_ylabel("CPU Utilization (%)")
    ax2[0, 1].grid(True, alpha=0.3)

    ax2[1, 0].plot(df["window_id"], df["memory_utilization_instant"], "g-")
    ax2[1, 0].set_title("Memory Utilization")
    ax2[1, 0].set_ylabel("Memory Utilization (%)")
    ax2[1, 0].set_xlabel("Time Window")
    ax2[1, 0].grid(True, alpha=0.3)

    ax2[1, 1].plot(df["window_id"], df["migrations_cumulative"], color="purple")
    ax2[1, 1].set_title("Cumulative Migrations")
    ax2[1, 1].set_ylabel("Migration Count")
    ax2[1, 1].set_xlabel("Time Window")
    ax2[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(out / "random_time_series.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # # 图3：风险对比
    # grp = df.groupby("active_servers_count")
    # avg_vm_risk = grp["vm_average_risk"].mean()
    # total_server_risk = grp["server_total_risk"].sum()

    # fig3, (axl, axr) = plt.subplots(1, 2, figsize=(16, 6))
    # bars1 = axl.bar(avg_vm_risk.index, avg_vm_risk.values, color="skyblue", edgecolor="black")
    # axl.set_title("Average VM Risk by Server Count")
    # axl.set_xlabel("Number of Active Servers")
    # axl.set_ylabel("Average VM Risk Value")
    # for b, v in zip(bars1, avg_vm_risk.values):
    #     axl.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.005, f"{v:.2f}", ha="center", va="bottom")

    # bars2 = axr.bar(total_server_risk.index, total_server_risk.values, color="lightcoral", edgecolor="black")
    # axr.set_title("Total Server Risk by Server Count")
    # axr.set_xlabel("Number of Active Servers")
    # axr.set_ylabel("Total Server Risk")
    # for b, v in zip(bars2, total_server_risk.values):
    #     axr.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.5, f"{v:.1f}", ha="center", va="bottom")

    # fig3.tight_layout()
    # fig3.savefig(out / "random_risk_comparison.png", dpi=300, bbox_inches="tight")
    # plt.close(fig3)


def main():
    """运行随机迁移方案，输出 random_results_detailed.csv 及配套图表。"""
    # 配置：与主流程保持一致的 VM/PM 数量
    VM_COUNT = 100
    PM_COUNT = 20

    # 初始化数据与控制器
    loader = DataLoader(vm_count=VM_COUNT, pm_count=PM_COUNT)
    loader.load_all_data()

    # 预测模型仅为满足控制器依赖，不做训练
    pred = PredictionModels(loader)
    controller = MPCController(loader, pred)

    # 计算时间窗口总数（采用各VM窗口数的最小值，保证覆盖）
    if loader.time_windows:
        window_counts = [len(w) for w in loader.time_windows.values()]
        time_steps = min(window_counts) if window_counts else 0
    else:
        time_steps = 0
    if time_steps <= 0:
        print("No time windows available.")
        return

    # VM/PM 集合
    vm_ids = sorted(list(loader.vm_pm_mapping.keys()))
    pm_ids = list(loader.pm_ids)

    # 生成随机迁移计划
    schedule = generate_random_schedule(
        time_steps=time_steps,
        vm_ids=vm_ids,
        pm_ids=pm_ids,
        seed=42,
    )

    # 执行并输出结果
    df = compute_random_results_with_controller(
        controller=controller,
        schedule=schedule,
        time_steps=time_steps,
        vm_ids=vm_ids,
        pm_ids=pm_ids,
        out_dir=".",
        csv_name="random_results_detailed.csv",
        consolidate_initial=True,
        initial_active_pm_ratio=0.2,
        initial_active_pm_count=None,
        normalize_by_total_pms=False,
    )

    # 绘图
    plot_random_figures(df, out_dir=".")


if __name__ == "__main__":
    main()