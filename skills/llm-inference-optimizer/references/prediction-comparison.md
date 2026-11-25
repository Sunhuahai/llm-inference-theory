# 逐轮事前预测与实际结果合同

每一候选用一个稳定 project/round ID（并行候选使用不同 ID）。round 指一次有明确干预和父版本的试验；同一轮的重复测量按预先冻结的统计量归约，原始样本由 evidence_refs 指向。不要将同一次试验的重复当成独立 holdout。

## 事前

参考 [prediction.json](../assets/prediction.json)。schema_version=1；project/round/theory_version/metric IDs 用 ASCII 字母数字开头，可含点、下划线和连字符，长度最多 80。parent_ref 是基线提交与配置身份；candidate_ref 是预先指定的候选身份（不必是尚未产生的提交）；context_id 是冻结的环境/工作负载/指标身份。mechanism、assumptions、validation_plan、theory_refs 必填。validation_role=search 或 holdout，后者必须含 shadow.version 和 shadow.points（与 active 点预测的相同指标）。

每个 metric 固定 level=resource/execution/service，unit、direction=lower/higher，statistic（如 mean、p99、total_tokens_per_sec），baseline 非负，baseline_tolerance_rel 非负小数。预测类型：

| kind | 必填参数 | 评价 |
| --- | --- | --- |
| point | value | 值误差与收益误差 |
| interval | low, high；可选 value 点预测 | 区间覆盖；有点才计算 MAE |
| lower_bound / upper_bound | value | 界是否满足，不计点预测误差 |
| unknown | reason | 记录实际值，不伪造预测 |

基线未知则先做基线观测，或在计划阶段明确缺口，不用零代替；零基线允许记录绝对量，无法定义相对收益。所有数值有限、非负；以 ms、bytes、tokens/s 等明确量纲，不在同指标混用。

冻结命令（参数中的目录须为新目录）：

```bash
python3 "$SKILL_DIR/scripts/skill.py" freeze optimization/rounds/round-001 /path/to/prediction-input.json
```

冻结保存 prediction JSON、Markdown 与 SHA-256、active/shadow manifest 摘要、时间。它能检测文件变化，不是可信时间戳/防恶意篡改系统；独立评价需确认真实采样时间在冻结之后。

## 事后

[measurement.json](../assets/measurement.json) 的五个身份字段与预测完全一致。status=completed/failed/blocked/inconclusive，correctness=passed/failed/unknown，comparability=comparable/conditional/noncomparable，evidence=measured/simulated/user_report/inconclusive。completed 要求 implementation_ref、evaluator_ref（真实执行版本和独立评价器身份）。evidence_refs 指原始日志/trace/配置及重复样本；不要把模拟当测量。每个实测 metric 对应已注册定义，包含 baseline（本次重新测量或可核验同口径基线）和 actual。缺失指标从 metrics 省略并说明。失败或阻塞也生成报告，不用 0 填缺失。

```bash
python3 "$SKILL_DIR/scripts/skill.py" compare optimization/rounds/round-001 /path/to/measurement-input.json
```

CLI 输出 result/measurement.json、comparison.json、comparison.md、receipt.json；目录不可覆盖。只有 completed、passed、comparable、有原始证据引用且基线漂移在冻结容忍度内的 metric 才可比较。simulated/user_report 可在本次报告作演示/报告比较，但不计实测累计统计或校准；simulated 禁止入用户库。

定义 lower 指标收益 g=(baseline-candidate)/baseline；higher 则 g=(candidate-baseline)/baseline。预测收益使用冻结 baseline，实测收益使用本次实测 baseline，记录相对基线漂移。误差 e=g_actual-g_pred，Markdown 以百分点报告；JSON 为小数比例。另记录实际值减预测值、区间命中、界满足。负误差是收益被高估；界未饱和不等于界错误。零 baseline 不计收益误差。

## Agent 的解释（必须）

同时阅读三层：物理量是否按预计变化、执行路径是否转化、服务指标是否转化。具体列资源计数器、执行时间线、队列或有效提交。明确支持、否定、未辨识和可能混杂；给出下一个最小区分办法。使用 [theory-update](../assets/theory-update.md) 与 [next-plan](../assets/next-plan.md)。理论更新不能覆盖原冻结文件；参见[理论演化](theory-evolution.md)。

历史汇总以同 theory version / context / metric / scope / unit / direction / statistic 分组，点预测 MAE 和 bias、收益是否正的分类正确率、区间覆盖率、界满足率分别统计。样本数、选择偏差和缺失率必须一起解读；不以少量已选择成功候选的结果证明总体泛化能力。
