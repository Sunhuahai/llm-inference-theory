# 提示词：根据已有优化结果分析下一步

将下面的提示词发给正在具体推理项目中工作的 Agent。它先查证项目历史，再按问题阅读本仓库，输出有证据的方向和最小验证计划。已有项不必重复填写；没有预先指定主目标或资源预算时，Agent 应给条件分析并指出决定性缺口。

[使用指导](../project-optimization-guide.md) · [全局图与分层子图](../../visualizations/diagrams/research-atlas/README.md) · [内容与实现状态](../content-map.md)

## 可直接复制

如果 Agent 已安装 [LLM Inference Optimizer Skill](../../skills/llm-inference-optimizer/README.md)，可直接调用 `$llm-inference-optimizer` 并说明 Review + Plan；下面的长提示词仍可供未安装 Skill 的 Agent 使用。

```text
请基于当前项目已经尝试的推理优化及其结果，分析下一轮最值得推进的优化方向。

理论参考仓库：Sunhuahai/llm-inference-theory
仓库地址：https://github.com/Sunhuahai/llm-inference-theory
本地参考仓库路径：如工作区已有副本请直接使用；否则通过当前已授权的 GitHub/文件访问方式阅读，并记录引用版本。
项目范围：当前项目；如上下文指定了目录、基线、实验集或硬件，以既有信息为准。
主目标、必须保持的条件与资源预算：先从现有项目文档及用户上下文提取。缺失时标 unknown，并说明它怎样影响方向选择，不擅自补充 SLO、dtype 或预算。

这次任务默认是只读分析和输出报告。保持已有用户授权范围：未要求实施时，不修改项目代码、不启动新 GPU 作业、不变更 oracle 或评分口径。不要仅给优化技术清单，也不要等待参考仓库具备完整 GPU optimizer 才开始分析。

一、建立可比较的项目事实
1. 读取当前项目维护规则、README、相关源码/配置、优化记录、提交和已有原始结果。查清固定原始基线、当前已确认最优、各候选 parent，以及修改是否实际生效。
2. 提取模型/精度/任务语义、设备与拓扑、引擎版本、负载与输出口径、缓存状态、主指标、质量要求、计时和重复方式。不要默认使用参考仓库历史 Qwen/H800 案例的条件。
3. 建立一行一个候选的台账：变更、parent、环境/负载、正确性、结果/不确定性、实际成本、来源、可比性、证据状态和接受决定。所有失败、超时和未决都保留。
4. 区分直接可比、仅在限定条件下可比和当前不可比。若 batch/调度/并行配置本身就是优化变量，以同任务分布、约束和总资源比较完整配置，不把它误报为同形状局部加速。
5. 未测或未找到的量写 unknown；有汇总但无原始日志时明确证据范围。能自行从已有文件恢复的信息直接查证，只把影响决策的缺口留给用户。

二、按问题阅读理论仓库
先读 docs/project-optimization-guide.md、docs/content-map.md，并用 visualizations/diagrams/research-atlas/README.md 定位全局关系。再只选与你发现的问题相关的章节：
- 慢因不明确：theory/diagnosis/README.md、optimizer/diagnosis/README.md。
- 局部改善未传到整体：theory/execution/README.md、models/execution/README.md。
- 性能边界与剩余机会：theory/performance/README.md、models/resource/README.md。
- 形状与算子效率：models/operator/README.md、experiments/microbenchmarks/README.md。
- 状态/缓存：theory/memory/README.md、theory/foundations/cache-context-ceilings.md。
- 投机生成：theory/generation/README.md。
- 吞吐、排队与尾延迟：theory/serving/README.md、models/serving/README.md。
- 资源作用机制与组合：theory/optimization/mechanisms.md、theory/optimization/coupling.md、optimizer/evolution/README.md。
- 预测失配：models/calibration/README.md、experiments/model-validation/README.md。
- 选择、成本与停止：optimizer/planning/README.md、optimizer/evaluation/README.md。
- 旧经验与跨环境复用：optimizer/memory/README.md、experiments/case-studies/README.md。
按需使用 docs/benchmark-protocol.md 和 experiments/templates/README.md 核对证据口径。
报告引用实际读过的文件及其相关判断；若参考仓库无法访问，明确说明，不能声称已经阅读。

三、从历史尝试提取机制证据
将每个技术名称翻译为它改变的计算量、分层流量、通信、容量、执行依赖、有效提交产出或调度条件，同时列新增开销与前置条件。
分别写“事实、当前解释、替代解释、未观测项”。检查功能是否原本已有，收益是否重复计算，当前 parent 是否使旧结果失去可比性。
编译错误不能否定整个机制；超时不是测到了零收益；无显著变化不能直接当成无效。负结果应限制具体适用域，并说明何时值得复核。

四、判断当前仍未辨识的瓶颈
对核心现象列出有证据基础的竞争假设。逐项写支持证据、反对证据、未观测量，以及什么观察会推翻它。
找出能让不同假设产生不同预测的最小区分办法。优先利用已有 trace、日志和源码；缺少信息时给出待执行小实验，写清固定条件、改变因素、观测指标、成本与预期分歧。
若多个解释对同一总延迟预测相同，不以重复同一个总延迟冒充可辨识性改善；若干预同时改变多个因素，明确混杂与归因边界。

五、提出下一轮优先事项
只保留最有依据的 2–3 项；证据只支持一项就给一项。允许首项是补测或恢复基线。
每项必须包含：
- 目标阶段/代码范围、当前 parent 和目标指标；
- 资源或依赖变化、前提、依赖/冲突；
- 净收益如何得到、包括什么新增开销、可能迁移到什么瓶颈；
- 项目证据与理论依据，明确严格界、近似预测、合成演示与未知项；
- 最小对照、正确性、测量/重复/留出计划；
- 实现与验证成本的估算来源，硬预算未知时明确标注；
- 接受、放弃、未决及重启条件；
- 为什么排在其他方向前面，以及什么新证据会改变排序。
不要编造具体加速百分比、置信概率或综合评分。低可信预测只作排序线索，不当硬约束。组合后重算资源与执行图，不能相乘独立加速比。文献成绩、参考仓库合成参数和其他硬件结果不替代当前项目证据。

六、按以下结构输出可交接报告
1. 当前判断：目标、当前最优、主要限制与最重要未知项。
2. 历史证据台账：可比性、事实、来源、失败及未决。
3. 机制层总结：已消除/未改变的成本、瓶颈迁移和需复核的旧结论。
4. 竞争假设与最小区分办法。
5. 下一轮优先事项及排序理由。
6. 验证与停止计划：固定对照、质量、全部成本、判据；标明哪些尚未执行。
7. 依据与关键缺口：项目证据位置、理论仓库文件及版本、最少待确认问题。

停止判断应针对当前方向：目标已满足、明示条件下剩余机会不足、预算不能覆盖验证，或缺少决定性信息。没有项目给定门槛时做条件判断，不发明门槛。几个失败不证明全局最优。
已有信息足以支持部分结论时，先完成这些分析；不要以非关键字段缺失为由只返回问题列表。
```

## 如何使用输出

先检查报告能否将每项建议追溯到项目证据与明确的作用机制，再按实际需求授权其中一个具体改动或小实验。已有授权持续有效，无需重复设置审批。运行后的结果可再次交给同一提示词，更新台账、竞争假设和排序，而不是覆盖原来的失败记录。

若需要评价“这套理论指导是否真的改善了 Agent”，另读[Agent Benchmark](../../experiments/agent-benchmark/README.md)，按相同工具、动作权限和总预算做对照；一次项目建议有效不能单独证明通用搜索优势。
