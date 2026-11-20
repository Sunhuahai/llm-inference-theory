# 第二轮实施任务书 · 三个大改方向

更新：2026-11-20（Asia/Shanghai）。目标：用户现有 Qwen3.8-Flash-Next 8-bit / Qwen4Exp + H800。用户已反馈 Week 1 全部尝试，要求更激进、模型特定的研究，并进一步要求写成本周任务卡。

**当前派发入口为 [R01–R03 任务卡](tasks/README.md)。** 本次将 GDN、PLE、MoE 三个大改方向拆成阶段任务；原 T016 HC 小 M 保留为短期独立实验。已有 T 编号和首轮历史保留，不把同一方向重复登记为新 trick。

## 可直接交给 Codex 的提示词

```text
接手 Sunhuahai/qwen38-next-h800-tricks 本周的三个大改方向。
先读取最新 main 的 AGENTS.md、environment.yaml、docs/model-and-support.md、
docs/benchmark-protocol.md、weeks/2026-W47/tasks/README.md，
以及本任务书和 R01/R02/R03 中实际选择的一张卡。

目标只用我的实际 Qwen3.8-Flash-Next-FP8/Qwen4Exp + H800。
Week 1 的 T001–T013 已全部尝试，先从已有部署找到 W1-best、实现commit、
环境快照和profile，不重做首轮。未知字段从当前运行环境读取，不按官方demo猜测。
如果我实际使用另一引擎或版本，映射实际模块并记录差异，不为匹配研究稿而换基线。

固定现有checkpoint及全部实际dtype、scale和计算精度，包括weight、activation、
KV/indexer、GDN state、卷积/PLE历史、新增ring/correction、dot math mode和累加。
保持原top-10、采样/grammar/logprobs、reasoning、EOS/输出要求和总GPU预算。
实现落在我的现有推理项目；研究仓库只保存文档、脱敏配置和结果。
只使用已有授权资源。若当前只有研究仓库，就交付代码适配方案和缺失入口，
不声称已经启动或完成GPU实验。

三项任务按以下范围推进，不能把所有阶段一次改完再报一个总数字：

R01：GDN状态重构与原生MTP，主线。
先做真实GDN shape的固定精度单层对照，再接分页、cursor/planner、prefill、
prefix/align flush、preemption和slot生命周期，然后独立接固定深度MTP。
普通decode仅为机制诊断，最终性能对照仍为W1-best。
核对上游PR正文和固定head代码的差异：不能照旧正文实现host cursor，
也不能仅按FP32 state标签忽略FP16 ring或tf32x3重建。
target/draft能力分别核对，PLE/QSA历史单独处理。
动态验证长度属于后续阶段；PLE/QSA/Indexer能力缺口修复前保持blocked，
不修改capability声明或删断言来跳过实现。

R02：PLE静态计算缓存，先做trace门槛。
公开host row-cache实验有真实文本近零重复的负结果；先用我的真实访问trace
分别测owner行命中、完整16-ID tuple命中、复用距离和实际暴露等待。
校准/评估trace分离；已有prefix cache消掉的工作不能再次算作PLE收益。
只有命中/成本/容量上界通过才写owner原字节热行cache，miss回原UVA；
投影cache是独立后续变体。想省lookup，probe必须早于现有prefetch。
保持唯一owner、精确键、原scale、ETP/HC-SP、MTP历史、Graph和失效规则。
trace不支持收益就交付parked结论，不为完成任务强写缓存。

R03：MoE内部分段与流水化，条件储备。
先查prepare/experts/finalize真实类、NoDPEP还是dispatcher、权重与scale布局、
top10路由、workspace别名、Graph和进程高水位。
先研究固定请求batch下的内部token分段，保留原专家kernel和外层调用边界；
有实际dispatch时再处理全rank一致容量和idle rank，不能丢路由或改变shared合并。
不启用Qwen4Exp当前不支持的DBO/ubatching，不把更换kernel与流水粒度混在一个A/B。
native MXFP8路径和重新缩放FC1的SM90 block-FP8路径都需按原精度合同排除；
名称含FP8不能代替权重字节/scale核对。

每张卡按阶段提交最小可回退commit，自定义开关标为拟议/项目内开关。
升级runner/engine/backend单独建对照；没有收益证据不扩大组合扫描。
采用同trace、同采样、同冷热状态和至少三次交错重复，测TTFT/TPOT分位数、
输出/总吞吐、stream gap、错误、显存和对应机制成本，保留实际输出长度。
草稿拒绝token不计吞吐；小kernel变快不能替代完整服务结果。

交付阶段commit、baseline/candidate IDs、配置diff、shape/dtype/math清单、
正确性证据、真实性能或明确阻塞、是否继续及回滚方式。
每完成一个阶段再决定下一阶段；缺触发条件时继续下一个可执行方向。
结果按本周results和feedback规范归档，未运行字段保持unknown。
```

## 当前范围与停止点

| 任务 | 现在先交付什么 | 什么时候进入后续开发 |
|---|---|---|
| [R01](tasks/R01-gdn-state-mtp.md) | 固定版本适配图、原状态与 replay 的精度/生命周期合同、单层对照 | 状态正确、机制有空间，且当前 runner 接口可实现 |
| [R02](tasks/R02-ple-memoization.md) | 独立 trace 的行/tuple 命中与成本加权上界 | cache overhead 与容量代价后仍有净收益空间 |
| [R03](tasks/R03-moe-pipeline.md) | 真实执行类和 workspace 生命周期、一个内部 chunk 的可行性设计 | 原 dtype/scale、路由与 collective 合同能保持，且有可释放成本 |

当前三项均为 `conditional`，R01 动态验证子阶段为 `blocked`。这表示研究任务已具体化，尚不表示用户环境已核对或 GPU 任务正在运行。实施负责人、仓库、运行 ID 和收益保持 `unknown`，由执行时填写。

[HC 小 M 融合 T016](tricks/T016-hc-small-m-fusion.md)是另外保留的短期实验。若选择它，就作为一个独立任务，不能加入上面三项的组合结果而不做对照。

## 版本、交付与组合

初筛代码固定 `12924016eb8dbaae87d5163a1b7ec2614a44a797`；本次深入核验 main 为 `aefea5c8e00c47032cda6d0b75704f6cd0a7c66d`，各 PR 另锁 head。详见[来源增补](sources-round-02.md)。这些是证据版本，不要求用户升级。

每项先独立对 W1-best，再在有效方案之间做组合。R01 ring、R02 cache、R03 workspace 可能竞争同一显存与带宽，不能乘各自局部倍数。若改变并发容量，先报告同工作量的资源变化，再单独验证容量带来的吞吐。

结果沿用[既有 CSV](../../templates/result.csv)，附 `<run_id>-summary.md` 和环境快照；新增机制诊断放 summary/JSON，不改统一表头。实验失败、支持缺口与无效方向同样保存。当前没有本轮 H800 实测。
