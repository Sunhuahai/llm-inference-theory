<!-- Generated seed snapshot; source: docs/project-optimization-guide.md @ 1b4c5a37763475eb39b31dd083fcad256fc2578e. Do not hand-edit. -->
# 用本仓库指导具体项目的下一轮优化

> 面向已经有代码、测量和若干优化尝试的项目 Agent。目标是根据已有证据，判断接下来该优化什么、先补哪项观测、何时应停止。本文是分析指导，使用它不依赖本仓库已经接通完整 GPU 优化器。

[全局关系图与分层子图](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/visualizations/diagrams/research-atlas/README.md) · [内容地图](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/docs/content-map.md) · [可直接使用的项目分析提示词](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/docs/prompts/review-project.md)

## 1. 从已有结果形成下一步判断

一个项目可能已经尝试融合、缓存、CUDA Graph 和不同并行配置，有的变快，有的没有收益。下一轮最有价值的问题是：**这些结果共同说明了什么，哪些慢因仍无法区分，还有哪种改动可能改变当前真正限制目标的因素？**

例如，一项减少访存的改动没有变快，至少允许几种解释：实际流量没有下降；流量下降但计算更慢；局部确实变快但不在服务关键路径；新增开销抵消了收益。仅记录“融合失败”会丢掉这些区别，也可能错误地排除后续机会。对应理论见[瓶颈诊断](../theory/diagnosis/README.md)和[优化机制](../theory/optimization/mechanisms.md)。

项目 Agent 的产出应是一份**有证据的方向判断**：哪些解释被支持、哪些仍未知、建议下一步做什么，以及什么结果会改变建议。默认先只读已有项目、日志和本仓库，完成分析报告。只有当前或既有用户授权包含具体实现或实验时，才进入相应范围；分析任务本身不要求启动新 GPU 作业、改代码或变更评分规则。

## 2. 先判断哪些历史结果可以一起解释

### 2.1 找到真实比较对象

从项目已有的 README、实验记录、提交、配置和原始输出中提取下表。能从文件查到的内容直接查证；未找到的保留 `unknown`，并说明是否影响结论。不要默认项目使用本仓库历史案例中的模型或硬件。

| 对象 | 必须明确的内容 | 为什么影响下一步 |
| --- | --- | --- |
| 目标与不变量 | checkpoint、实际 dtype、任务/采样语义、质量要求、主目标、SLO | 决定哪些收益属于同条件优化 |
| 实现身份 | 固定原始基线、当前已确认最优、各候选的 parent SHA、配置和真实后端 | 区分累计收益与当前边际收益，检查已有优化是否重复 |
| 运行环境 | 设备与卡数、拓扑、引擎/编译器版本、资源隔离 | 防止把环境变化归因于补丁 |
| 工作负载 | 输入/输出长度分布、到达 trace、并发、缓存状态、请求完成口径 | 判断实验是否处在同一工作点 |
| 测量与正确性 | 计时边界、预热、重复、oracle、误差容限、原始记录 | 判断数值差异是否足以支持结论 |
| 已付出成本 | profiling、实现、编译、失败、复测、GPU/LLM/墙钟消耗 | 决定下一项信息或收益是否值得获取 |

若 batch、并行方式或调度本身就是待优化变量，应比较**相同任务分布、服务约束和总资源下的完整配置**，不强行要求这些变量相同；也不能把配置变更后的单请求数字解释为同形状 Kernel 加速。详细口径见[测量协议](benchmark-protocol.md)、[候选契约](../optimizer/search-space/README.md)和[实验模板](../experiments/templates/README.md)。

### 2.2 建立一行一个候选的证据台账

推荐字段如下。候选可以有多次重复记录，不能只保留最好的一次。

```yaml
trial_id: 项目中的稳定标识
parent_ref: 父版本与配置
fixed_baseline_ref: 固定比较基线
context_ref: 环境和负载记录
change: 实际生效的修改，不只是开关名称
mechanism_claim: 事前或事后解释，二者分别标记
correctness: passed / failed / unknown
measurement_refs: [原始结果、日志、分析脚本]
metric_scope: kernel / device_step / request / service
result: 指标、重复和不确定性，未知项保留 unknown
cost: 已知实际消耗与尚缺记录
comparability: comparable / conditional / noncomparable
evidence_status: hypothesis / derived / literature / simulated / user_report / measured / validated / inconclusive
decision: accepted / rejected / inconclusive / blocked
```

`comparable` 表示足以支持所声称的比较；`conditional` 表示只在明确子集或假设下可比；`noncomparable` 表示当前不能组成有效收益判断。后两类仍可保留作为实现线索，不应丢弃或偷偷补齐。证据、实现状态和候选决定是不同维度，见[研究方法](methodology.md)。

先处理会改变方向判断的缺口。例如只有 profiler 下的耗时、基线 SHA 缺失或输出长度变化时，优先恢复比较口径；缺少与当前判断无关的计数器，不必阻塞整份分析。对于只有汇总表的结果，可引用它说明作者/用户报告了什么，同时明确缺少原始证据。

## 3. 把每次尝试翻译成机制证据

将技术名称展开为“改变了什么 → 为什么可能影响目标 → 代价和条件是什么”。统一资源表示可参考[机制文档](../theory/optimization/mechanisms.md)：计算量、分层流量、通信量、容量轨迹、执行依赖、每轮有效提交数与质量约束。

| 已尝试方向 | 应追问的机制 | 一次无收益结果仍能留下什么 |
| --- | --- | --- |
| 融合或 tile/布局调整 | 哪层流量减少？是否增加寄存器压力、spill 或额外指令？ | 区分未触发新路径、资源未改善、或改善被其他瓶颈遮蔽 |
| 前缀/状态缓存 | 命中什么状态？节省多少工作？恢复和驻留成本是多少？ | 限定复用适用域，发现容量或恢复成本门槛 |
| CUDA Graph / launch 优化 | 消除了多少暴露间隙？是否增加 padding 或 fallback？ | 限定可消除开销，识别关键路径是否转移 |
| 通信重叠或并行重配 | 消息、rank 等待、依赖与争用分别怎样变化？ | 区分通信量大、晚到、不可重叠与资源竞争 |
| 投机生成 | 有效提交数增长是否覆盖 draft、verify 和状态成本？ | 限定适用长度/接受率范围，检查生成语义 |
| batch / 调度调整 | 改变的是执行效率、等待、缓存还是阶段资源比例？ | 识别吞吐、尾延迟和容量的权衡 |

对每条历史结果分别填写：**事实、当前解释、替代解释、未观测项**。编译失败说明该实现未成立；超时说明在分配成本内未完成；正确但变慢说明某工作点净收益为负。它们都不自动否定整个机制。分类及复用规则见[经验记忆](../optimizer/memory/README.md)。

原始基线上的收益不能直接充当当前最优版本上的收益。若 A 已经消除了某段开销，B 的边际机会可能变小；A 也可能释放另一资源，使此前无效的 B 变得有用。用 parent 绑定条件收益，并阅读[优化耦合](../theory/optimization/coupling.md)与[代码演化](../optimizer/evolution/README.md)。

## 4. 根据具体疑问选择阅读路径

先看[总图](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/visualizations/diagrams/research-atlas/README.md)定位问题，再读最相关的理论、模型和验证章节。报告中说明“这篇文档帮助判断了什么”，不要求每次通读仓库或罗列所有链接。

| 项目中的实际疑问 | 优先阅读 | 读完应作出的判断 |
| --- | --- | --- |
| 总延迟或利用率相似，却说不清慢因 | [诊断理论](../theory/diagnosis/README.md) → [Agent 诊断接口](../optimizer/diagnosis/README.md) | 列出竞争解释，选择能让预测分歧的观测 |
| 局部 Kernel 变快，端到端几乎不变 | [执行与并行](../theory/execution/README.md) → [执行图模型](../models/execution/README.md) | 检查暴露关键路径、调用占比和瓶颈迁移 |
| 想知道某方向最多还能省多少 | [性能边界](../theory/performance/README.md) → [资源模型](../models/resource/README.md) | 区分条件上限、可校准预测与未知可达性 |
| 大 shape 有效，小 shape 失效 | [算子模型](../models/operator/README.md) → [微基准](../experiments/microbenchmarks/README.md) | 检查形状分段、固定开销与有效速率外推 |
| 缓存命中增加，但性能或容量没有改善 | [内存与状态](../theory/memory/README.md) → [缓存专项](../theory/foundations/cache-context-ceilings.md) | 区分命中口径、状态恢复、驻留和服务影响 |
| 投机/MTP 收益不稳定 | [生成理论](../theory/generation/README.md) → [候选机制](../theory/optimization/mechanisms.md) | 比较有效提交产出和完整周期成本 |
| 吞吐上升但 TTFT/p99 退化 | [服务理论](../theory/serving/README.md) → [服务模型](../models/serving/README.md) | 检查到达、排队、动态 batch 和 SLO 权衡 |
| 两项有效优化组合后不及预期 | [耦合理论](../theory/optimization/coupling.md) → [组合验证](../experiments/model-validation/README.md) | 区分重复节省、顺序依赖、竞争与迁移 |
| 预测总能拟合基线，却选错候选 | [校准与不确定性](../models/calibration/README.md) → [模型验证](../experiments/model-validation/README.md) | 检查干预差值、排序、可辨识性与留出 |
| 候选很多，时间和 GPU 有限 | [规划与搜索](../optimizer/planning/README.md) → [独立评价](../optimizer/evaluation/README.md) | 比较净收益、信息价值、成本和停止条件 |
| 旧环境的成功或失败能否复用 | [经验记忆](../optimizer/memory/README.md) → [跨环境案例](../experiments/case-studies/README.md) | 区分机制、代码与参数迁移，明确再校准成本 |
| 想证明理论指导比普通 Agent 更有效 | [Agent Benchmark](../experiments/agent-benchmark/README.md) | 保持动作/工具/预算可比，设计理论与记忆消融 |

[解析参考](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/models/analytic/README.md)和[场景预测器](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/models/predictor/README.md)仅在项目能提供相应输入且模型假设成立时辅助计算。缺失数据不能用示例参数代替；CPU 合成输出不能变成项目实测结论。[离线搜索回放](../optimizer/systems/README.md)可帮助理解预算和选择逻辑，但不是运行项目分析的前置依赖。

## 5. 维护竞争假设，并寻找最小区分实验

对目前最重要的现象，保留少量有实际依据的竞争假设。若证据只支持一个，也应说明排除了哪些解释；不要为凑表格编造竞争原因。

| 假设 | 已有支持/反对证据 | 它对一次干预的预测 | 最便宜的区分办法 |
| --- | --- | --- | --- |
| H1：某资源成本限制目标 | 指向具体计数器、shape 或时间线 | 改变相应工作量后目标应怎样变化 | 优先重读现有 trace；不足时提出受控小实验 |
| H2：另一依赖或成本遮蔽收益 | 指向执行图、等待或其他路径 | 同一干预可能只改变局部时间 | 对比暴露时间，保持测量边界一致 |
| H3：收益被新增成本抵消 | 指向恢复、spill、同步或管理开销 | 去掉/限制新增成本后结果应变化 | 选能分离机制与代价的最小变体 |

表中的 H1–H3 是组织方式，不是通用瓶颈标签。真正选择实验时应写明：固定什么、只改变什么、观测什么、各解释预期有何不同、什么结果会让它们都失效。不能近似隔离变量时，明确混杂因素，降低因果结论强度。

**优先消除结构歧义，再用重复降低噪声。** 如果两种假设总预测相同延迟，多测几次同一个总延迟只会缩小误差条，无法区分原因。也可能存在更经济的选择：已有两个实现可直接比较，或一个低成本候选本身就能充当区分实验。理论依据见[可辨识性与反事实诊断](../theory/diagnosis/README.md)。

### 合成例：一次“没有加速”怎样改变下一轮方向

以下数字全部为教学构造，不代表任何项目实测。简化模型为 $T=\max(T_C,T_M)$；暂时假设两项干预没有额外开销。

| 已有记录 | 条件资源时间 | 总时间 | 目前支持的判断 |
| --- | --- | ---: | --- |
| 基线 | 计算 10 ms、访存 8 ms | 10 ms | 当前计算约束更强 |
| A：减少计算 | 计算 6 ms、访存 8 ms | 8 ms | A 后可能转向访存约束 |
| B：减少访存 | 计算 10 ms、访存 5 ms | 10 ms | B 在原基线上无净收益；尚未否定其条件价值 |
| A+B | **尚未评价** | unknown | 需要新的预测与证据 |

在上述模型内，A+B 预测为 $\max(6,5)=6$ ms。但若 A 后存在未观测的 8 ms 调度/依赖下限，联合修改可能仍为 8 ms；若融合增加开销，还可能更慢。下一步应围绕这两个解释检查 A 的时间线，或在组合成本很低且语义可验证时比较 A 与 A+B。不能直接把 6 ms 填为已取得结果。

这里 B 的负结果阻止了“在原基线上继续投入同类访存改动”，同时保留了“计算瓶颈解除后重新评估 B”的机会。建议是否先采集时间线还是直接做组合，取决于现有证据、组合成本和剩余预算；没有这些输入就给条件建议，不编造精确优先级。

## 6. 把下一轮方向写成可审核的候选

每个推荐方向至少回答以下问题。通常只保留最有依据的 2–3 项，证据只够一项时就给一项。

| 字段 | 必须写清的内容 |
| --- | --- |
| 作用对象 | 当前 parent、目标阶段、代码/配置范围、目标指标 |
| 机制与前提 | 要减少或改变的资源、依赖、容量、有效产出，以及已确认/未知的条件 |
| 净收益预测 | 由什么模型得到，包含哪些新增成本，可能出现什么新瓶颈 |
| 证据与不确定性 | 项目文件/日志、读取的理论章节、假设和未辨识项 |
| 最小验证 | 参考实现、工作点、正确性、对照、观测、重复或留出方式 |
| 搜索成本 | 已知估算来源、最小资源需求、硬上限；未知时说明如何确定 |
| 接受与放弃条件 | 哪种观察支持收益，哪种否定机制，何时记为未决 |
| 交互关系 | 是否依赖既有改动、与谁重复/冲突、是否需要重测组合 |

预测净收益应在完整变更模型中计算：

$$
\widehat{\Delta T}_a=T_{\rm parent}-\widehat T_a.
$$

这里 $\widehat T_a$ 包含变更后的依赖、资源竞争与新增成本。不能把多个重叠分项的节省直接相加，也不能把两个独立加速倍数相乘。只有在明确假设下，才能把某个暴露成本的完全消除当作乐观上限；由有效速率计算的估计仍是预测，见[理论界与可达性能](methodology.md)。

### 如何决定先后

先排除明确违反质量、语义、支持性或硬资源约束的方案。其余按以下理由说明顺序，而不是给没有数据基础的百分制评分：

1. **修复关键证据缺口：** 如果比较不可用会改变所有后续结论，先恢复基线或计时口径。
2. **低成本区分主要解释：** 一项观测能明显改变后续选择时，把它作为候选动作。
3. **实施有依据的净收益：** 结合当前关键路径、质量风险、可复用实现和验证成本，而不只看局部理论节省。
4. **保留适量探索：** 近似模型预测差或旧环境曾失败，不等于确定不可行；需要说明为何现在值得重新检查。

只有收益/可行性分布和成本有依据时，才使用[成本感知搜索](../optimizer/planning/README.md)中的期望改进。资料不足时使用排序理由、情景范围和“若……则……”条件建议；不把主观把握写成统计概率。

### 何时停止或暂缓

停止条件针对当前任务和方向，不宣称全局最优：主目标已满足且其他约束通过；在明确假设下剩余机会低于预先定义的有意义改善；预算不足以完成实现加独立验证；关键输入不可取得，无法区分主要解释；或新增实验的信息/预期收益不足以覆盖成本。只有用户或项目已给门槛时才能据此判停，不能自行编造 SLO 或“至少提升 5%”规则。

一次超时、一次不显著或连续几个失败不能单独证明方向无效。记录停止理由、适用范围和重启条件，例如版本改变、新 profile、容量释放或上游瓶颈被消除。正式接受要求见[独立评价](../optimizer/evaluation/README.md)。

## 7. 项目 Agent 的固定输出结构

报告应能由下一位 Agent 继续使用，采用以下结构，并链接实际项目证据：

1. **当前判断。** 用简短段落说明目标、当前最优、主要限制与最重要的未知项。
2. **历史证据台账。** 列出尝试、parent、结果、可比性和来源；区分有效、失败、未决与不可比。
3. **机制层总结。** 已消除什么、未改变什么、瓶颈是否迁移、哪些旧结论需要复核。
4. **竞争假设。** 支持/反对证据、预测分歧及最小区分办法。
5. **下一轮优先事项。** 每项用第 6 节的候选字段，说明排序理由和条件；覆盖必要的补测动作。
6. **验证与停止计划。** 固定对照、质量口径、全部成本和判据；把已做的分析与尚未执行的实验分开。
7. **依据与缺口。** 项目证据位置、本仓库实际阅读的路径及所用判断；仅列影响决策的待确认信息。

已有内容足以形成条件结论时，先完成报告；不要因为非关键字段缺失就停止。无法访问关键项目数据时，明确访问边界并提出最少的针对性问题，避免将一般性建议写成已完成项目诊断。

完整可复制提示词见[项目优化复盘与下一步分析](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/docs/prompts/review-project.md)。它把本页的方法、[全局图](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/visualizations/diagrams/research-atlas/README.md)、[理论](../theory/INDEX.md)、[模型](../models/README.md)、[Agent 方法](../optimizer/README.md)和[实验](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/experiments/README.md)连接为同一个使用入口。
