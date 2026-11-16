# T004｜合并 QSA 投影并核对 SM90 调度

| 字段 | 内容 |
|---|---|
| ID / 优先级 | T004 / P1 |
| 状态 | `researched`；待 H800 实测 |
| 调研日期 | 2026-11-16 |
| 固定条件 | checkpoint、所有 dtype、并行布局、采样与输出要求 |

## 问题与机制

QSA 的主投影和 indexer 投影可合成一次 GEMM，减少小矩阵场景的重复调度。[vLLM PR #59533](https://github.com/vllm-project/vllm/pull/59533) 合并 QKVG 与 indexer Q/K，并列出 H200 SM90、TP4、小 M 的调度支持。其展示的 GB300 微基准不代表 H800 服务加速。

## 适用与版本门槛

需要包含该 PR 的 Qwen4Exp NVIDIA 路径。核对实际 GPU 架构、TP、矩阵形状和 dispatch；PR 中的 LoRA 路径仍使用独立投影。基线已经合并时标 `baseline-covered`，不要重做相同实现。

## 交给实现 Agent

| 步骤 | 操作与交付 |
|---|---|
| 1 | 固定引擎 SHA，检查 `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` 与模型调用，导出实际 M/N/K 和 kernel 名称。 |
| 2 | 在同一 build 内构造独立投影/合并投影对照，保持权重布局转换可逆，不改任何 dtype。 |
| 3 | 先测投影输出、stride 和边界；再分别测小 batch decode 与较大 prefill。 |
| 4 | 若另调 SM90 参数，单独建实验，与投影合并分开归因；交付最小 patch 和 profiler 摘要。 |

## 单变量 A/B

第一轮只切换投影合并，第二轮才对合法调度参数做 A/B。并发 1/4/16、固定短 prompt 与长输出用于观察 TPOT；8K/32K prompt 用于观察 TTFT。报告 p50/p95、output tokens/s、总 tokens/s，以及 GEMM 耗时；仅在显存允许的工作点运行。

## 正确性与回滚

检查合并后的 Q/K/V/gate/indexer 输出与原路径在既定误差阈值内一致，再运行相同业务样例。误差超限、尾延迟退化或吞吐收益不稳定则恢复原投影/调度。不得调整输出长度掩盖退化。

## 收益已知与未知

上游确认实现已合并，但当前 H800 是否命中优化形状、收益大小均未知。必须同时给出局部与端到端结果，不外推上游微基准加速比。
