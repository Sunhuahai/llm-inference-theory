# T011 — CPU 调度、NUMA 与异步执行

> **已归档（2026-11-20，Round 01）。** 生命周期：`archived`；尝试记录：`user_report / attempted`。用户明确 T001–T013 均已尝试。逐项收益、质量与实施证据尚未提交，保留为 `unknown`；下文的研究状态和“待实测”措辞是原始历史记录。不得据此重新派发，也不得推断全部成功或失败。
> 当前新增候选见 [Round 03](../../../rounds/round-03/README.md)；归档说明见 [Round 01 索引](../README.md)。

- ID：T011
- 状态：researched
- 调研日期：2026-11-16
- 验证：待 H800 实测

## 问题与机制

GPU 空洞有时来自 tokenizer、scheduler、网络发送或 CPU 争用。vLLM 的异步调度把 CPU 下一步调度与 GPU 当前步执行重叠；它是否有收益取决于实际 CPU 开销。[架构说明](https://docs.vllm.ai/en/latest/design/arch_overview/) NVIDIA 建议结合 GPU/NIC 的 NUMA 位置检查进程与内存亲和性。[NCCL 指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/performance_and_tuning.html)

## 模型与版本门槛

冻结 FP8 checkpoint、全部 dtype 和 PLE 放置/offload 策略。[官方模型配方](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) 的最低 vLLM 为 0.30.0，80GB Hopper 示例用 PLE CPU offload，因此还要记录 host 内存/传输压力。保存 CPU quota/cpuset、NUMA、commit 与 resolved flags；异步调度/MTP/structured output 组合仍需独立核验。

## 实施 Agent 步骤

1. profile API、tokenizer、scheduler 和 GPU launch gap；确认压测客户端自身未成为瓶颈。
2. 在既有 CPU 资源内调整 worker/API 分配，或按 GPU 邻近 NUMA 绑定；尊重容器 cpuset，不把多个忙进程挤入同一核。
3. NUMA 验证后另开实验切换异步调度。固定 streaming 间隔，避免用缓冲更多 token 制造表面吞吐提升。[当前参数说明](https://docs.vllm.ai/en/stable/configuration/engine_args/)

## 单变量 A/B

A 为原 CPU/NUMA 或异步配置，B 只改其中一项。固定 GPU 数、请求到达轨迹、并发、输入输出、缓存及流式发送粒度。报告端到端 TTFT/TPOT、ITL、输出 tok/s、CPU 使用与调度等待、GPU 空洞；服务端和客户端测量分开标注。

## 正确性与回滚

检查流式顺序、取消/超时、多请求状态隔离及固定质量集；不修改宿主机 BIOS、全局 governor 或隔离策略。回归或功能不兼容时恢复原进程绑定/异步参数。

## 已知与未知

已知：CPU 争用可能影响 GPU 服务。未知：本机是否 CPU-bound、可用核心与版本兼容性；GPU 已饱和时不能期待此项必然有效。
