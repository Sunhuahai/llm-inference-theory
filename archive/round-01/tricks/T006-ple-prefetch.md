# T006｜PLE 异步预取与 CPU / GPU 放置

> **已归档（2026-11-20，Round 01）。** 生命周期：`archived`；尝试记录：`user_report / attempted`。用户明确 T001–T013 均已尝试。逐项收益、质量与实施证据尚未提交，保留为 `unknown`；下文的研究状态和“待实测”措辞是原始历史记录。不得据此重新派发，也不得推断全部成功或失败。
> 当前新增候选见 [Round 03](../../../rounds/round-03/README.md)；归档说明见 [Round 01 索引](../README.md)。

| 字段 | 内容 |
|---|---|
| ID / 优先级 | T006 / P1 |
| 状态 | `researched`；待 H800 实测 |
| 调研日期 | 2026-11-16 |
| 固定条件 | checkpoint、PLE 与所有其他 dtype、并行布局、输出要求 |

## 问题与机制

该模型包含大型 N-gram / PLE 查找表。[Qwen 官方说明](https://github.com/QwenLM/Qwen3.8-Flash-Next) 支持将表放在 host RAM，并通过异步预取与模型计算重叠。目标是减少查表传输等待，同时留下 GPU 工作空间；收益取决于真实内存与互联瓶颈。

## 适用与版本门槛

[vLLM recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) 给出 `--engram-config '{"cpu_offload":true}'`，说明默认开启，并在 80GB H100 配方中依赖它。H800 需核查卡数、每 rank 显存和 NUMA；不能用总体显存推断能关闭 offload。已具备异步预取时标 `baseline-covered`，再定位剩余等待。

## 交给实现 Agent

| 步骤 | 操作与交付 |
|---|---|
| 1 | 固定引擎 SHA，记录表实际 dtype、放置、CPU 内存峰值、PCIe / NUMA 路径和预取实现。 |
| 2 | 抓取查表传输与计算时间线，确认等待是否位于关键路径。 |
| 3 | 优先单独比较 CPU/线程 NUMA 绑定；若继续修改预取时序或缓冲区，另开独立 A/B。 |
| 4 | 只有逐 rank 容量验证通过才比较 GPU resident/offload；不够则记录不可行，不能改精度凑容量。 |

## 单变量 A/B

每次只改放置、绑定或预取策略中的一项。固定请求内容，测试长 prompt 与持续 decode，分别记录冷/热阶段。报告 TTFT/TPOT p50/p95、output tokens/s、总 tokens/s、host/GPU 峰值内存、PCIe 流量及 GPU 等待时间。

## 正确性与回滚

核对查表索引、行内容、跨请求 buffer 生命周期，覆盖并发和重复 token。出现错行、竞争、内存不足或尾延迟退化，恢复原放置与绑定配置，回退预取 patch。

## 收益已知与未知

官方支持该机制，但用户 H800 是否受 PCIe、CPU DRAM 或 NUMA 限制尚未知。offload 可是部署前提，也可能增加延迟；不能把节省显存直接写成性能提升。
