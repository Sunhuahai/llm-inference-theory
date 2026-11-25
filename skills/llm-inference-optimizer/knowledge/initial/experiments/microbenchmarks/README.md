<!-- Generated seed snapshot; source: experiments/microbenchmarks/README.md @ 1b4c5a37763475eb39b31dd083fcad256fc2578e. Do not hand-edit. -->
# 微基准：校准执行成本与适用边界

> 状态：采集设计。尚未新增真实 GPU 测量或硬件参数表。

微基准回答“这一类操作在指定形状和资源条件下花多少时间”。系统收益还取决于调用频率、关键路径与服务调度，不能直接把局部加速比乘到端到端结果上。

## 1. 从真实形状抽样

先从目标引擎收集调用 shape、dtype、布局、后端和调用频率，再选择主工作点、临界区与留出点。不要仅测容易达到峰值的大矩阵。

| 类别 | 自变量 | 记录的结果 | 关键反例 |
| --- | --- | --- | --- |
| GEMM / grouped GEMM | M/N/K、专家 token 分布、stride、后端 | 设备耗时、实际流量、工作区 | 小 M 与不均衡专家不能由大矩阵效率外推 |
| Attention | q/kv 长度、head 结构、batch、分页与布局 | Prefill/Decode 分开、缓存状态 | GQA/MLA/混合模型不能统一代入普通 MHA 成本 |
| 状态访问 | 物理页、共享比例、容量压力、恢复路径 | 读写流量、恢复时间、峰值容量 | 逻辑共享不保证实际只读一次 |
| Collective | 消息大小、rank 数、拓扑、协议 | 每 rank 时间、暴露通信与尾部 | 总集群流量除单链路带宽会错算 |
| Runtime | kernel 数、Graph 命中、padding/fallback | host 提交间隙、设备完成时间 | 累计 API duration 不等于关键路径开销 |

## 2. 计时与诊断分开

稳态测量先完成编译、加载与预热；冷启动成本单列。设备计时和客户端端到端计时分开，异步执行需要明确同步边界。记录独占资源、时钟/功耗设置、背景负载、种子和缓存准备方式。

计数器采集单独运行。Nsight Compute 的 replay、cache 控制与采集可能改变执行环境，不能把这次墙钟时间当作正常服务结果。[官方 Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)解释了相关机制。对通信、并发 kernel 和 CPU/GPU 重叠，保留时间线与资源条件。

## 3. 数据如何进入模型

计时输出应包含 `environment_id`、`workload_id`、`operator_signature`、`repeat_id`、`duration_ms`、`timing_scope`、原始证据路径与状态。不同 dtype、路径和 shape 区域使用不同参数；没有依据的统一“70% 峰值效率”不能覆盖整个模型。

有效速率 $F/T$ 或 $D/T$ 可帮助描述工作点，但不证明该资源主导，也不是物理容量上界。用一个点反算参数再预测该点是校准，不是验证。每个校准簇留出不同形状，按 [模型验证](../model-validation/README.md)检查迁移与干预误差。

## 4. 第一组实验如何选

选择一个能独立验证、对实际系统贡献明确的算子；测基线、一个机制清楚的候选、一个可能触发失效的形状。为这个小集合写出可证伪判断，例如“减少中间流量的收益在大上下文更明显，但小 shape 时被固定开销限制”。只有结果支持后再扩展 sweep。

完整环境与结果用 [模板](../templates/README.md)记录。真实 GPU 命令需要目标机的软件/工具版本及访问条件，本文不提供假定已安装的启动参数。其后接入 [算子模型](../../models/operator/README.md)与 [系统案例](../case-studies/README.md)。
