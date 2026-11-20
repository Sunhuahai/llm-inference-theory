# Round 01 · T001–T013 已尝试归档

归档日期：**2026-11-20（Asia/Shanghai）**。用户原话：“01-013 都尝试过了，归档。再找 10 个 trick”。

本次归档改变的是任务生命周期：**从当前待执行清单移入历史**。证据来源为 `user_report`，尝试状态为 `attempted`；每项的成功/失败、性能、质量、实现 commit 和运行附件仍为 `unknown`。不能把这 13 项批量标为已验证、失败或无收益。

原路径保留跳转，历史卡片完整保留；本周 [feedback](../../weeks/2026-W47/feedback/2026-11-20.md) 与 [results](../../weeks/2026-W47/results/README.md) 仍是结果入口。将来收到结果时补到相应历史卡片和结果文件，不重新编号。

| ID | 历史任务 | 生命周期 | 尝试证据 | 逐项结论 |
|---|---|---|---|---|
| T001 | [分块预填充与批调度](tricks/T001-prefill-scheduling.md) | archived | user_report / attempted | unknown |
| T002 | [让 CUDA Graph 覆盖真实 Decode 形状](tricks/T002-cuda-graphs.md) | archived | user_report / attempted | unknown |
| T003 | [T003｜融合 QSA 准备阶段](tricks/T003-qsa-prepare.md) | archived | user_report / attempted | unknown |
| T004 | [T004｜合并 QSA 投影并核对 SM90 调度](tricks/T004-qsa-projection.md) | archived | user_report / attempted | unknown |
| T005 | [T005｜GR / PLE 序列并行](tricks/T005-gr-ple-sequence-parallel.md) | archived | user_report / attempted | unknown |
| T006 | [T006｜PLE 异步预取与 CPU / GPU 放置](tricks/T006-ple-prefetch.md) | archived | user_report / attempted | unknown |
| T007 | [按容量与拓扑选择并行布局](tricks/T007-parallel-topology.md) | archived | user_report / attempted | unknown |
| T008 | [混合模型的前缀 Checkpoint 与两池分配](tricks/T008-hybrid-prefix-cache.md) | archived | user_report / attempted | unknown |
| T009 | [T009｜原生 MTP 深度与负载适配](tricks/T009-native-mtp.md) | archived | user_report / attempted | unknown |
| T010 | [MoE 通信后端与计算重叠](tricks/T010-moe-communication.md) | archived | user_report / attempted | unknown |
| T011 | [CPU 调度、NUMA 与异步执行](tricks/T011-host-runtime.md) | archived | user_report / attempted | unknown |
| T012 | [后置候选：Prefill / Decode 分离](tricks/T012-prefill-decode-disaggregation.md) | archived | user_report / attempted | unknown |
| T013 | [上下文压缩与 Prefix 复用的协同优化](tricks/T013-context-compression.md) | archived | user_report / attempted | unknown |

## 历史与当前入口

- [首轮研究原文](research.md) · [首轮任务书原文](agent-tasks.md)。它们记录当时的顺序，不再作为默认执行命令。
- [Round 03：新增 T021–T030](../../rounds/round-03/README.md)。
- [Round 02：T014–T020 与 R01–R03](../../weeks/2026-W47/research-round-02.md)。用户未明确说这部分已经尝试，本次不替用户归档。
- [参考 skill](../../skills/optimize-qwen-h800/SKILL.md)：实施 Agent 先读取 skill，再读取选中的一张现行任务卡。

归档前内容可按仓库 commit [`915023a`](https://github.com/Sunhuahai/qwen38-next-h800-tricks/tree/915023a373d3b7d2a6726df02e53804b74dfc3e8/weeks/2026-W47) 核对。周号仍按真实日期记录；实验轮次独立递增。
