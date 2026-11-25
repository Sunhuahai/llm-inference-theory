<!-- Generated seed snapshot; source: models/contracts.md @ 1b4c5a37763475eb39b31dd083fcad256fc2578e. Do not hand-edit. -->
# 性能模型的数据契约

> 版本 1，2026-11-25。下列为跨模块的概念契约；当前 `analytic` 与历史 `predictor` 各有已实现的窄 schema，不能将本页全部字段声称为已支持接口。

## 1. 输入：场景、资源、干预、证据

| 对象 | 必需信息 | 约束 |
| --- | --- | --- |
| TaskSpec | checkpoint/tokenizer/采样、精度、功能、质量口径 | 候选共享同一语义签名；变语义另开研究轨道 |
| HardwareSpec | 资源 ID、速率、单位、峰值/有效类型、来源和版本 | 多 rank/链路独立标识；未知参数不得设为 0 |
| WorkloadSpec | trace hash、到达、长度联合分布、缓存、负载模式 | 统计窗口和实际发送口径固定 |
| ExecutionSpec | 节点、工作量、依赖、资源顺序、内存生命周期 | 图必须无环；删边必须有执行依据 |
| InterventionSpec | 基线 SHA、变更机制、成本/图/容量变化、必要前提 | 必须可从原始基线重建候选 |
| EvidenceSpec | 来源、artifact/hash、采集工具、时间、split | 文献、合成、用户报告、实测分别标识 |

工作量用 FLOP、byte、operation；速率用对应单位每秒；报告时延可用 ms，但转换只有一次。MB/GB 与 MiB/GiB 需在边界显式转换，不能在 JSON 中靠读者猜。

实际项目可从[环境、负载与试验模板](../experiments/templates/README.md)准备输入；候选的父版本、机制、前提与冲突按[候选表示](../optimizer/search-space/README.md)登记。

## 2. 输出：区分五种语义

| `value_kind` | 可以表达什么 | 不能推断什么 |
| --- | --- | --- |
| `resource_dependency_lower_bound` | 已给工作/速率上界/依赖下的必要时间 | 可达性或真实延迟 |
| `fixed_schedule_lower_bound` | 附加固定资源顺序下的必要时间 | 任意重排/新算法的普适下界 |
| `fixed_schedule_model_estimate` | 给定有效速率和排程的时间估计 | 严格性能极限或实测 |
| `dependency_model_estimate` | 忽略未建模争用的图时间估计 | 完整系统完成时间 |
| 外部 `measured` 证据 | 独立评估器执行得到的原始观测 | 模型自动证明正确或跨域适用 |

输出还应保存 `input_origin`、假设、适用范围、缺失机制、容量可行性、预计瓶颈和运行版本。代码运行获得一个推导值，并不把 `synthetic` 输入升级成真实测量。

## 3. 基线与候选的可比性

比较使用同一 TaskSpec、硬件预算、负载/统计窗口和测量目标。候选可以改变图与资源分配，但必须披露。只有 $T_0$ 是实际时间且 $L_1$ 是候选严格时间下界时，$T_0/L_1$ 才是相应条件下的加速上界。两个下界 $L_0/L_1$ 只表示边界变化；两个预测时间之比只表示模型预测。

如候选违反容量、正确性或服务约束，输出不可行及原因，不因速度预测好就纳入最优候选。对于未知的质量检查，输出未知，不能自动当通过。

## 4. 观测误差接口

解析参考支持可选 `observations`，记录 `scenario_id`、`duration_ms`、`origin`、`split` 和 `artifact_ref`。它只进行数值对照，不采集数据、不训练参数、不独立验证原始证据。`split=holdout` 也只是输入声明，真正的留出集访问隔离由[实验协议](../experiments/model-validation/README.md)负责。

预测模式报告有符号相对误差；下界模式检查观测是否低于输入条件声称的下界。后者是矛盾诊断，出现时应检查输入和假设，而不是调高预测以掩盖问题。

## 5. 与 Agent 交换什么

建议模型响应返回四类信息：可行/不可行/未知；有明确语义的数值；影响结果的关键假设；最值得补充的测量。搜索控制器可以使用这些信息排序和设计实验，但只有独立正确性与服务评价能接受候选。经验记录同时保存预测前版本与实测后版本，防止事后修改预测造成“总是预测正确”。

[可运行 schema](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/models/analytic/README.md) · [历史预测器 schema](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/models/predictor/SCHEMA.md) · [校准](calibration/README.md) · [自主优化](../optimizer/README.md)
