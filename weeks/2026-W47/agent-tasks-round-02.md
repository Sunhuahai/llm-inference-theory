# 第二轮实施任务书 · Qwen4Exp / H800

日期：2026-11-20。目标仅为用户现有 Qwen3.8-Flash-Next 8-bit 部署，仓库暂定官方制品 `Qwen/Qwen3.8-Flash-Next-FP8`。用户已完成首轮尝试，允许测试性的大幅代码改造；逐项原始结果尚未归档。

## 可直接交给 Codex 的提示词

```text
接手 Sunhuahai/qwen38-next-h800-tricks 的第二轮模型专属优化。
先读最新 main 的 AGENTS.md、environment.yaml、docs/model-and-support.md、
docs/benchmark-protocol.md，以及 weeks/2026-W47 下：
model-audit-round-02.md、research-round-02.md、本任务书和对应 trick 卡。

这次只针对我的实际 Qwen3.8-Flash-Next-FP8/Qwen4Exp + H800。
Week 1 的 T001–T013 我已全部尝试，先找到现有最佳有效配置、运行commit和profile，
不要重新从初始13项做起；缺的环境字段从当前部署读取并归档，不凭官方demo补猜。

沿用现有checkpoint和全部实际dtype，包括激活、weight scale、KV/indexer、
GDN checkpoint、卷积/PLE历史、新增correction/replay buffer及累加精度。
不换模型、不量化、不剪枝/蒸馏、不缩短输出、不关闭reasoning或采样功能，
不扩大已有GPU总预算。研究仓库不预造另一个推理引擎，代码落在当前实施项目。

当前研究的vLLM证据固定SHA为12924016eb8dbaae87d5163a1b7ec2614a44a797；
若我用的是别的版本或SGLang，先映射实际同类模块，不为匹配文档升级后混淆基线。
所有新代码使用独立开关，能回退到原路径。工作只使用已获授权的服务器和资源；
若只有研究仓库，就先完成代码适配方案并准确记录执行入口缺失。

本批最多实施以下三项，按profile触发，单项通过才组合：

1) T016：Qwen4Exp HC小M上投影+门控融合。
   查真实M分布和原kernel。现有HC已含PDL，不能把“打开PDL”算新收益。
   对原小M路径与仅改变fused-up适用阈值做A/B；若现有fused kernel不适合H800，
   再独立写SM90小M kernel。最终mixer的down→SiLU作为单独变体。
   数值、scale、累加语义不变，测完整HC链及端到端增量。

2) T017：GDN固定精度checkpoint/replay执行。
   针对36层、48 value heads、128×128 state的实际TP分片，先做单层。
   不照抄SM100/FP16 FlashInfer开关；保持现有state与全部buffer/correction精度，
   核查论文代码里FP16 correction缓存是否违反本机原精度。
   先teacher-forced logits和每种接受长度后的状态等价，再接Qwen4Exp注册、
   QSA scratch、PLE卷积、prefill/prefix/preemption/slot生命周期。
   普通decode与原有固定深度MTP分开对照，不同时改验证策略。

3) T020：PLE精确缓存。
   保留目前的prefetch、ETP、HC-SP和lookup输出shape。
   先在owner内建立只读热行HBM缓存，缓存原字节，miss回原UVA；
   必须维持一个global row只有一个有效owner，不能把多个rank的缓存字节相加。
   用独立trace验证命中，统计真实暴露等待和state池容量损失。
   只有热行试验完成后，再独立尝试精确16个row IDs对应的[k,v]投影缓存；
   hidden-dependent gate和short-conv继续正常执行，不缓存它们。

T014、T018、T019是储备，除非本批方向不满足触发条件，不并行扩成大扫描。
T015在所审计vLLM路径被PLE和QSA/Indexer能力检查阻塞，不盲目开
enable_adaptive_verification，也不跳过能力检查伪装可用。
Qwen4Exp MegaMoE路径会将block128 FP8转成MXFP8，不能作为保持现有精度的方案。

每项用首轮最佳有效配置为baseline，报告新增收益；不得回到弱eager基线。
单变量、固定工作负载/采样/冷热状态/总GPU、至少三次交错重复。
报告真实TTFT/TPOT/输出tokens/s/总tokens/s、错误率、显存、机制profile和质量；
草稿拒绝token不计吞吐，尾延迟与失败不能从统计里丢掉。

交付每项实现commit、开关/配置diff、baseline/candidate IDs、环境快照、
dtype/scale清单、实际shape、正确性与性能结果、是否保留及回滚方式。
全部实测与负结果归档到本周results/和feedback/，不伪造成功数字。
一个方向被阻塞时把代码证据写清，继续下一个可执行方向，不要求我重新确认常规可逆工作。
```

## 先复用的首轮证据

只补必要信息：最佳运行的 model revision/engine SHA、卡数与进程组、所有实际 dtype、baseline ID、固定负载、一次阶段 profile。若当前工作目录已有这些，不重复采集同一组数据。

不能推断“全部尝试”意味着全部成功或失败，也不能在无结果文件时给 T001–T013 批量标 validated。用户报告已归档为 [user_report](feedback/2026-11-20.md)。

## 三项各自的最小交付

| 任务 | 第一阶段 | 进入完整模型的条件 | 不值得继续时 |
|---|---|---|---|
| T016 | 一份真实小 M HC 链 A/B | 同精度正确、链路时间改善超过波动 | 原阈值已覆盖或新融合使小 GEMM变慢，保留负结果 |
| T017 | 真实 GDN shape 的 replay/flush/accept-state 验证 | dtype完整一致、状态生命周期正确，且I/O有净节省 | SM90实现未能保精度或flush尾延迟抵消平均收益 |
| T020 | owner内不可变热行cache + 独立trace回放 | miss正确、实际等待减少、容量代价可接受 | lookup早已完全隐藏、命中低或占用state池导致吞吐回退 |

主场景加一个回归场景即可完成首轮筛选。更广的组合/负载扫描只在解释具体剩余风险时做。

## 组合顺序与结果口径

先各自对 W1-best 做单变量对照，确定两个胜者后再做组合；组合对照使用其中一个胜者为基线，另外保留 W1-best 锚点。不能把局部倍数相乘。

T016 改计算组织，T017 改 GDN state 数据流，T020 使用 HBM 缓存；三者可能竞争带宽和显存，组合收益需要实测。若未来加入 T014，保留 MTP target verification 的 logits 及 sampler功能；若未来完成 T015 适配，无 confidence head 的接受估计仍需要 draft logits。

结果文件沿用 [既有CSV表头](../../templates/result.csv)，附 `<run_id>-summary.md` 和环境快照。新增诊断可放 summary/JSON 附件，不临时破坏统一 CSV。尚未跑就写 unknown，不写示例速度。
