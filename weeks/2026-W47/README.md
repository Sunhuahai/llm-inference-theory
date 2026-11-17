# 2026-W47 · 建立可信基线，筛选第一轮优化

日期：**2026-11-16—2026-11-22**（Asia/Shanghai）。调研完成日期：2026-11-16。

**当前没有 H800 实测数据。** 已建立 13 张候选卡片、测量协议与实施任务。部署卡数、框架、业务 trace 和基线仍待提供。

## 本周最值得先做的事

1. **P0 环境与基线**：确认模型、H800/拓扑、实际 dtype、版本，查已有融合路径；产出可比较的 baseline。
2. **P1 调度与 Graph**：依据 baseline 分别做 T001 或 T002 单变量试验；优先短/长请求混合时的尾延迟。
3. **P2 模型专属瓶颈**：若 QSA/GR/PLE 耗时明显，优先 T003–T006；业务重复前缀多则优先 T008。先有证据再选择，不一次性改全套。

**新增研究入口（2026-11-17）**：[前缀缓存命中、有限状态容量与上下文压缩的收益上界](../../theory/cache-context-ceilings.md)，附可运行合成回放与质量 gate；T008/T013 的真实收益均待实测。仅在真实长前缀重复或业务冗余充分时提升它们的优先级。

直接交付入口：[实施任务书](agent-tasks.md)。完整研究判断：[research.md](research.md)。

## 候选登记

| ID | 卡片 | 状态 | 决定先后顺序的条件 |
|---|---|---|---|
| T001 | [Prefill 调度](tricks/T001-prefill-scheduling.md) | researched | 混合请求或 TTFT/TPOT 冲突 |
| T002 | [CUDA Graph](tricks/T002-cuda-graphs.md) | researched | launch/CPU 间隙、短 batch decode |
| T003 | [QSA prepare 融合](tricks/T003-qsa-prepare.md) | researched | 先查当前 build 是否覆盖 |
| T004 | [QSA 投影融合](tricks/T004-qsa-projection.md) | researched | 小 GEMM 与实际 SM90/TP 形状 |
| T005 | [GR/PLE sequence parallel](tricks/T005-gr-ple-sequence-parallel.md) | researched | 支持的多卡路径，长 prefill |
| T006 | [PLE 预取与 offload](tricks/T006-ple-prefetch.md) | researched | host 取表/PCIe/NUMA 等待 |
| T007 | [并行与拓扑](tricks/T007-parallel-topology.md) | conditional | 合法多卡配置，固定总预算 |
| T008 | [混合前缀缓存](tricks/T008-hybrid-prefix-cache.md) | researched | 多轮 continuation 或前缀分叉 |
| T009 | [原生 MTP](tricks/T009-native-mtp.md) | researched | 接受率、验证开销与质量门槛 |
| T010 | [MoE 通信](tricks/T010-moe-communication.md) | conditional | 专家不均衡或 collectives 占比高 |
| T011 | [Host runtime](tricks/T011-host-runtime.md) | researched | CPU/分词/主机调度瓶颈 |
| T012 | [Prefill/decode 分离](tricks/T012-prefill-decode-disaggregation.md) | conditional | 多机规模、真实流量、混合状态传输支持 |
| T013 | [上下文压缩](tricks/T013-context-compression.md) | conditional | 业务确有冗余、压缩成本与严格配对质量门槛 |

## 记录入口

- [反馈说明](feedback/README.md)
- [结果说明](results/README.md)
- [决策记录](decisions.md)
- [一手来源索引](sources.md)

首周无 carry-over。每晚更新只补新增反馈和明确下一步；尚未运行的候选不写收益数字。
