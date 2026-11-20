# 任务交付模板

按以下字段写一份短报告。允许写 unknown，但必须说明缺什么；没有运行则省略性能表数据行。结果放实际运行所属周，任务卡保持原 ID。

```text
任务：T0xx / 阶段 Sx
研究卡片：路径与文档 commit
实施仓库：已授权仓库
基线：run_id、环境ID、代码SHA、配置/数据hash、原始结果引用
候选：patch commit、唯一变量、默认开关与回滚方法

事实：已观察到的源码/日志/profile（标来源）
假设：为什么这个变量可能改善当前瓶颈
门槛：卡片前置条件、是否已内建、实际shape/dtype/backend
精度：所有dtype、scale/cast/math模式是否保持；对应审计证据
正确性：执行了什么、预先容差、结果和失败样例引用
资源：相同总GPU预算；HBM/host/通信新增成本
性能：局部机制指标；三组交错端到端结果与波动；错误/质量/输出长度
状态：baseline-covered / conditional / blocked / unsupported / inconclusive /
      regressed / validated（未测只能写研究/实现状态，不写validated）
证据来源：literature / user_report / agent_measured / hypothesis
下一步：只给一个最小动作，或说明当前结束
回滚：具体开关/commit恢复、缓存或Graph需重建的条件
```

## 报告句式

- 未实测：`已完成 T0xx 的源码定位与设计；缺少实际 H800 环境/基线附件，GPU 正确性与性能未运行。状态 blocked。`
- 已内建：`实际 build 的函数 A 已执行卡片目标机制，证据为日志/源码 B。标 baseline-covered，不重新实现。`
- 负结果：`局部算子更快，但包含 pack/通信的端到端结果落在基线波动内。保留原路径，状态 inconclusive。`
- 成功：`在相同环境、负载和资源下完成三组交错对照；质量与合同通过，收益超过基线波动。详见原始数据与实现 commit，状态 validated。`

这些句子是表达模板，不是已经发生的结果。不要复制示例中的状态作为实测结论。

## experiment-plan 字段怎样填

| 字段 | 填写规则 |
|---|---|
| task_id / card_path | 一张现行 T 卡；路径相对研究仓库，不填 R 工作流整体 |
| implementation_repo | 实际已授权推理仓库；不要填本研究仓库来代替 |
| baseline_run_id | 已有最佳有效配置的真实运行 ID；引用 evidence.baseline |
| candidate_commit | 实施仓库候选补丁的完整 commit；prototype 可 unknown，benchmark 前必填 |
| single_variable | 只留一个条目，写清实际 before/after；多阶段分别建计划 |
| engine_base_sha | 两组共同的原引擎完整 SHA；候选 commit 另填上项 |
| weight_scale_contract / accumulation_contract | 原 scale 格式与数值、cast/reduction 次序和累加模式；引用 source_audit 的检查结果 |
| graph_mode / mtp_mode | 实际 Graph 模式、MTP 是否启用/K/verify 合同；把本卡刻意修改的唯一机制写在 single_variable，不能偷偷改其他模式 |
| sampling_contract / output_stop_contract | 实际采样、penalty、EOS/stop/长度合同；不能只写“默认” |
| thinking_mode / feature_contract | 实际 reasoning 与工具/grammar/logprobs 等所需功能 |
| load_contract / cache_mode | 相同并发或到达轨迹、冷/热缓存条件、预热口径 |
| chat_template_sha256 / workload_sha256 | 对实际使用的模板和数据/trace manifest 计算完整 SHA-256，不用示例 hash |
| gates | 只有相关证据存在且满足卡片条件才填 true；未查保持 false |
| evidence | 指向已读的真实日志/源码/结果引用；一个路径字符串不构成自动验证 |

invariants 两侧记录的是本次**应保持不变**的合同。若发现其他字段也在变化，拆实验或纠正计划，不删除字段绕过检查。数值模板与合成检查样例不得写入真实性能结果。
