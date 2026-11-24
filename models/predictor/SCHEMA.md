# 预测器输入契约（schema_version = 1）

执行示例：`python3 models/predictor/predict.py models/predictor/examples/synthetic-baseline.json --format json`。

根对象有五组字段：`schema_version`、`data_kind`、`baseline`、`profile`、`scenarios`。请从 [合成 JSON 示例](examples/synthetic-baseline.json) 复制结构，逐项替换，不要从未知硬件参数凭空猜数。

## baseline：比较的固定起点

| 字段 | 单位 / 约束 | 解释 |
|---|---|---|
| `id` | 字符串 | 脱敏环境与 workload 标识 |
| `ttft_mean_ms` | ms，正数 | **低负载均值**，同一批客户端请求的到首 token 时间 |
| `tpot_mean_ms` | ms，正数 | **低负载均值**，有效输出 token 的平均时间 |
| `saturated_output_tps` | output tok/s，正数 | 固定 workload 下扫并发测得的饱和输出容量 |
| `stage_mean_ms.prefill` | ms，正数 | 与 TTFT 同请求集合的 Prefill 均值 |
| `stage_mean_ms.first_decode` | ms，正数 | 首个目标 token 的 decode 均值 |
| `decode_tokens_per_step` | 整数，MTP 情景需要等于 1 | 防止重复计算已启用的 speculative decode |
| `resource_busy_fraction` | {资源名:0–1} | **饱和窗口**中相对忙碌比例，资源名必须与候选因素一一对应 |
| `arrival_rps`、`avg_output_tokens` | 可选非负 / 正数 | 若填写，另给按到达负载限制的可交付吞吐上界 |

约束：`TTFT ≥ prefill + first_decode`；差值作为未归属的固定 TTFT（如 tokenizer、网络与空队列开销）。如差值显著偏离真实 timeline，说明阶段定义不一致。

**资源标签并不自动等于真实瓶颈**。建议先从 GPU Compute、HBM、互联和 CPU 等真实可度量资源入手。观察窗口必须与饱和基线一致；CPU/GPU 测量都可能有 profiler 扰动。

## profile：分阶段、互斥的执行账

三个必填列表：`prefill`、`first_decode`、`decode`。每个 block：

| 字段 | 单位 / 默认 | 含义 |
|---|---|---|
| `name` | 每阶段唯一字符串 | 如 qsa、moe、host、ple |
| `serial_ms` | ms，默认 0 | 暴露的串行片段 |
| `compute_ms` / `comm_ms` | ms，默认 0 | 可并行计算与通信 lane 的时间 |
| `overlap_fraction` | 0–1，默认 1 | 两条 lane 实际隐藏较短 lane 的比例 |
| `compute_floor_ms` / `comm_floor_ms` | ms，默认 0 | 可实证支持的物理/实现下界，不得高于基线 lane 时间 |

每个 block 必须有非零成本。每阶段所有 block 的时间**不能重复计费**：例如已经计入 GPU 执行的时间，不能再按 CPU 调用耗时完整加一遍。

如果只能得到一个粗粒度阶段，可先用一个 block 占满该阶段，但这只能预测粗粒度改善，不能分辨具体算子的收益。

## scenarios：候选方案与不确定区间

每个情景至少有唯一 `name`，可加 `assumption` 说明保守、中心或乐观。`effects` 是可选的 block 变化数组，每个元素：

| 字段 | 解释 |
|---|---|
| `stage` / `block` | 准确对应 profile 中某一个 block；同一 block 只能出现一次 |
| `serial_speedup`、`compute_speedup`、`comm_speedup` | 大于 0，默认 1；小于 1 可表达退化 |
| `overlap_fraction` | 覆盖这一 block 的新重叠比例，0–1 |
| `added_ms` | 实际暴露新增开销（ms），非负，**不跟着校准比例缩放** |

`resource_demand_factors` 另设一组映射，且**必须与 baseline.resource_busy_fraction 的所有资源名完全一致**。因子是“候选每输出 token 的需求 / 基线每输出 token 的需求”。缺失这组因素时输出吞吐容量为 `null`，不能简单从 TTFT/TPOT 比例推得。

估计建议：计算用每输出 token 的相关 GPU busy / FLOP demand，HBM 用读写字节需求，通信用 bytes 和 collective 时间，CPU 用暴露的 CPU demand；**都须按同一 workload 的最终输出 token 归一**。基于 profile 推出的因素标为 hypothesis，单独留出实测校准。因素不会由 `effects` 自动产生，因为延迟关键路径和容量资源需求并非同一个量。

### speculation（可选）

只有 baseline 一次 target step 产出一个 token 时，候选可以填写：

| 字段 | 含义 |
|---|---|
| `draft_tokens` | 1–32 的整数 |
| `acceptance_probability` | 每步条件接受概率 0–1 的简化假设 |
| `verification_multiplier` | 一轮验证成本相对于优化后普通 decode step 的倍数 |
| `extra_per_round_ms` | draft / 控制 / 同步的额外毫秒 |

使用 `E[K] = 1 + p + … + p^d`；真实非独立接受长度应进一步替换为实测均值。当前 `speculation` 不自动改变 TTFT，也不推导资源需求因子，避免误导。

### observed（可选）

有真实 A/B 后，在对应情景中填 `observed`：

~~~json
{
  "ttft_mean_ms": 135.0,
  "tpot_mean_ms": 18.0,
  "saturated_output_tps": 2050.0
}
~~~

**这里只演示字段，不代表任何 H800 结果。** 输出会为已预测的指标计算带符号百分比误差。预测器输出 `warnings`、`floor_hits`、`capacity_bottleneck`、`expected_tokens_per_verification`，用于判断需要补充哪一种证据。

## 不同层级应如何使用

- **目前：** 只用合成数据验证公式与软件的内部一致性。
- **有真实基线：** 输入 profiler 与饱和压测，给每个 trick 单独建立三档参数情景。
- **有真实 A/B：** 填 observed，比较不同负载的误差，修正独立的参数。
- **有足够多测量：** 才能评估留出集误差，并考虑 trace-level 排队模型或残差学习。

不要把模型假设写成概率置信区间，也不要用本模型直接宣称 p95/p99、复杂调度下的 loaded TTFT 或 P/D 分离收益。
