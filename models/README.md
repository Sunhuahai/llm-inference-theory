# 性能建模与预测

> 2026-11-25｜理论驱动自主优化的第二支柱。专题模型和数据契约已建立；两个可运行的 CPU 参考用于校验条件性关系。尚无跨模型/硬件校准库或新的 GPU 实测结果。

模型的任务是把“这个优化减少了什么成本”转化为可复核的数值判断。一个合格的输出同时包含**值、量纲、适用域、证据类型、假设和未知项**；无法支持的尾延迟、吞吐或质量指标不填猜测值。

## 1. 五层体系与横向校准

[查看性能模型关系子图](../visualizations/diagrams/research-atlas/03-models.svg) · [返回全局图集](../visualizations/diagrams/research-atlas/README.md)。子图同时展示资源下界与有效预测、执行和服务的反馈，以及现有 CPU 参考的覆盖范围。

| 层 | 输入与问题 | 输出与章节 |
| --- | --- | --- |
| 资源 | 算法、形状、状态、硬件上限；必须完成多少工作？ | FLOPs、各级访存、通信与容量：[资源账本](resource/README.md) |
| 算子 | 实际形状、精度、后端及资源工作量；局部实现需要多久？ | 条件性下界或校准时间曲面：[算子建模](operator/README.md) |
| 执行 | 数据依赖、stream/资源顺序、并发争用；哪些成本暴露？ | 关键路径与干预后的图：[执行图](execution/README.md) |
| 服务 | 到达、长度、缓存、路由、调度；请求会得到什么服务？ | 请求时间线、容量和 SLO：[服务模型](serving/README.md) · [仿真设计](simulation/README.md) |
| 决策 | 基线/候选的可比输出；下一次实验值得做什么？ | 收益、误差、适用范围：[收益预测器](predictor/README.md) · [自主优化](../optimizer/README.md) |

[模型校准](calibration/README.md)贯穿各层。[统一契约](contracts.md)规定它们怎样交换数据，避免同名指标实际上使用不同时间窗口、硬件边界或任务语义。

先读直观例子：某节点需 2 ms 计算与 8 ms 显存访问。在理想重叠模型中，下界为 8 ms；消除大部分流量后，计算可能成为新的 2 ms 限制。若节点后还有 4 ms 的必要传输，系统下界从 12 ms 变为 6 ms。只有这些输入确为必要工作和资源上界时，数值才是条件性下界；它们不是 GPU 已经测到的延迟。

## 2. 当前可运行内容

| 组件 | 已实现 | 明确边界 |
| --- | --- | --- |
| [analytic](analytic/README.md) | 单位检查、多资源账本、依赖图/固定顺序、候选图重算、观测误差接口 | 不自动解析模型、不推导显存峰值、不模拟任意资源争用 |
| [predictor](predictor/README.md) | 基线校准的阶段情景、条件 MTP 成本、归一化资源需求代理 | 需要用户输入局部收益与资源因子；不是自动学习的通用预测器 |
| [cache_context](predictor/CACHE_CONTEXT_SCHEMA.md) | 已给定精确键 trace 的按序 LRU 回放 | 不支持并发分页/异构状态池，不输出在线 p99 |
| 分布式与服务仿真 | 本次补齐接口、事件规则和验证设计 | 完整实现及实测校准待后续研究 |

从仓库根目录运行，均只使用 Python 标准库：

```bash
python3 models/analytic/analyze.py models/analytic/examples/synthetic-system.json
python3 models/predictor/predict.py models/predictor/examples/synthetic-baseline.json --format markdown
python3 -m unittest discover -s models/analytic/tests -v
python3 -m unittest discover -s models/predictor/tests -v
```

示例均标为 `SYNTHETIC`。运行成功证明代码按指定模型计算，不证明模型准确预测真实硬件。

## 3. Agent 怎样消费模型

Agent 提供候选改变的成本与假设，模型返回资源可行性、条件收益、预计瓶颈和适用域。独立评估器随后返回实际观测，校准层分析误差。候选局部加速不能仅由 Agent 主观填写后再被模型“证实”；关键参数必须来自推导、校准或明确的情景范围。预测不确定时，下一步可以是补测，而不是立即改代码。

[理论总纲](../theory/foundations/unified-theory.md) · [模型验证](../experiments/model-validation/README.md) · [业界建模工具](../research/industry/infersim-aiconfigurator.md)

已有具体项目与历史尝试时，先按[项目复盘指南](../docs/project-optimization-guide.md)整理证据与未决问题，再选择本页对应模型层。模型输入应来自项目证据，输出应能帮助决定实现、补测或停止。
