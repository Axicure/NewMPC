# 基于MPC的虚拟机调度系统

这是一个基于模型预测控制（Model Predictive Control, MPC）的虚拟机调度系统，用于优化虚拟机在物理机上的分配，以降低安全风险、减少资源争用、降低功耗并最小化迁移成本。

## 系统架构

系统由以下几个主要模块组成：

1. **数据加载模块 (data_loader.py)**: 负责加载虚拟机和物理机的数据，并按时间窗口聚合数据。
2. **预测模块 (prediction_models.py)**: 使用ARIMA模型预测CPU使用率，使用LSTM模型预测内存使用率。
3. **MPC控制器模块 (mpc_controller.py)**: 实现MPC算法，优化虚拟机放置以最小化总成本。
4. **主程序模块 (main.py)**: 协调各个模块的运行，并可视化结果。

## 运行环境

- Python 3.7+
- 详细依赖请参见 `requirements.txt`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保数据文件已正确放置：
   - `data/container_machine_id.csv`: 虚拟机和物理机的对应关系
   - `data/mac_keys/50.csv`: 物理机ID列表
   - `data/target/instance_*.csv`: 虚拟机性能和风险数据

2. 运行系统：

```bash
python main.py
```

3. 查看结果：
   - 程序执行后会生成 `mpc_results.png` 图表和 `mpc_results.csv` 数据文件
   - 日志保存在 `mpc_scheduler.log`

## 参数调整

在 `main.py` 中，你可以调整以下参数：

- `VM_COUNT`: 虚拟机数量
- `PM_COUNT`: 物理机数量
- `START_WINDOW`: 开始时间窗口
- `END_WINDOW`: 结束时间窗口
- `PREDICTION_STEPS`: 预测步数

在 `mpc_controller.py` 中，你可以调整MPC参数：

- `alpha`, `beta`, `gamma`, `delta`: 各项成本的权重
- `risk_threshold`: 风险评级阈值
- `max_servers`: 最大服务器数量
- 其他参数

## MPC目标函数

MPC的目标是最小化以下加权成本函数：

C(t) = α · C_Cr(t) + β · C_Pc(t) + γ · C_Mc(t) + δ · C_Rc(t)

其中：
- C_Cr(t) 表示集群的综合共居风险
- C_Pc(t) 表示功耗
- C_Mc(t) 表示虚拟机迁移成本
- C_Rc(t) 表示资源争用程度

## 结果分析

系统运行后会生成三种图表：
1. 成本变化曲线
2. 每个时间窗口的迁移次数
3. 攻击者数量变化曲线

这些图表可以帮助分析MPC优化的有效性和系统安全性。 