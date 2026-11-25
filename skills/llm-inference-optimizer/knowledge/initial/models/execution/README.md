<!-- Generated seed snapshot; source: models/execution/README.md @ 1b4c5a37763475eb39b31dd083fcad256fc2578e. Do not hand-edit. -->
# 执行图模型：把局部成本组合成完成时间

> 状态：`derived`，2026-11-25。[CPU 参考](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/models/analytic/README.md)实现显式依赖和受限的固定资源顺序，不实现任意 GPU 并发仿真。

## 1. 两种图回答不同问题

**数据依赖图**只表示“B 必须等 A 的结果”。两个无依赖节点仍可能抢同一 GPU。**固定排程图**还包含 stream、设备和资源的先后约束。只有明确了资源分配与节点时长，最长路径才可作为这个排程的完成时间模型。

对于 DAG 上节点时长 $\tau_v$，最早完成时间递推为：

$$
e_v=\tau_v+\max_{u\in\mathrm{pred}(v)}e_u,\qquad T_{\rm graph}=\max_v e_v,
$$

无前驱时最大值取 0。使用每节点严格下界时得到图对应的下界；使用有效速率预测时得到条件模型估计，不能更名为严格下界。

## 2. 共享资源不能凭空并发

只有数据依赖时，还需考虑总资源需求下界：

$$
T\ge\max\left(T_{\rm path},\max_r\frac{\sum_v W_{v,r}}{R_r^{\max}}\right).
$$

两独立节点各在同一资源上需要 4 ms，最长路径只有 4 ms，总资源需求给出至少 8 ms。它仍可能不是紧下界，因为节点内部与其他资源的依赖还会限制调度。

当前参考程序的 `fixed_schedule` 是一个有意收窄的模型：凡共享任一资源的两个正工作量节点，必须在图上有先后顺序；否则拒绝计算并要求使用 `dependencies_only` 或补充串行边。不同资源的节点可并行。实际 concurrent kernels 使用“部分资源份额”时，应扩展为资源受限调度并重新校准，不能强行视为各自独占。

## 3. 一个完整的合成例

Attention 的资源下界为 8 ms。完成后 Projection 需要 3 ms，独立链路 Transfer 需要 4 ms，二者并行。因此该固定图的下界为 $8+\max(3,4)=12$ ms。

| 干预 | Attention | Projection | Transfer | 图下界 |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 8 | 3 | 4 | 12 ms |
| 减少 attention 流量 | 2 | 3 | 4 | 6 ms |
| 再减少传输 | 2 | 3 | 1 | 5 ms |

最后一步中传输单独减少了 3 ms，但系统只减少 1 ms，因为关键路径变成 Projection。两步都必须在候选图上重算。12/6 只表示两个下界之比，不能宣称实测 2× 加速。

## 4. 干预不仅改变节点时长

融合会合并节点、消除中间访存，且可能引入新 workspace；重叠会改变边，且可能改变有效时长；并行配置会改变每 rank shape 和集合通信。候选应提交完整的新图或可复核的图变换。一个固定 `speedup=2` 无法表达这些效应。

资源依赖、容量与图必须使用同一场景版本。比较前固定工作量、模型和精度，记录候选改变了哪些边及原因。删除一条边必须有语义或执行依据，不能为了预测变快而删除真实依赖。

## 5. 观测与验证

时间线应解释 launch、等待、copy、collective 的先后关系，而不是直接求和 profiler duration。[Nsight Compute][profile]的采集和 replay 会扰动运行；正式 wall time 用独立运行测量。验证先检查单节点，再检查路径、最后检查服务结果。若候选出现预测外的串行等待，更新图和速率，保留原预测作误差证据。

[执行理论](../../theory/execution/README.md) · [统一契约](../contracts.md) · [模型验证](../../experiments/model-validation/README.md)

单次执行图通过检查后，将阶段耗时传给[服务模型](../serving/README.md)；涉及动态 batch、队列与缓存生命周期时，按[事件仿真设计](../simulation/README.md)重建状态过程。

[profile]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "NVIDIA Nsight Compute Profiling Guide; 核对于 2026-11-25"
