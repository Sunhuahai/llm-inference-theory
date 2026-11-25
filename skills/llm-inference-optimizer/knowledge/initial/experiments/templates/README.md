<!-- Generated seed snapshot; source: experiments/templates/README.md @ 1b4c5a37763475eb39b31dd083fcad256fc2578e. Do not hand-edit. -->
# 实验记录契约与模板

这里的 JSON 是可复用的记录模板，不是已完成的实验，也不是解析模型或搜索回放的 CLI 输入。未知值用 `null`，不填 0；具体运行前必须补齐必要字段。实现级输入契约见 [Models contracts](../../models/contracts.md) 与 [Optimizer search space](../../optimizer/search-space/README.md)。

需要把多轮记录转成后续优化方向时，按[项目指导](../../docs/project-optimization-guide.md)组织可比性、机制解释与竞争假设，并使用[项目 Agent 提示词](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/docs/prompts/review-project.md)生成可交接分析。

| 文件 | 作用 | 哪些字段运行前必须确定 |
| --- | --- | --- |
| [environment.json](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/experiments/templates/environment.json) | 版本、精度、设备与拓扑 | checkpoint/revision、引擎 SHA、资源及实际 dtype |
| [workload.json](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/experiments/templates/workload.json) | 请求分布、到达、缓存、采样与 SLO | 数据 hash、输出口径、负载方式、约束 |
| [trial.json](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/experiments/templates/trial.json) | 一个候选从预测到评价的记录 | parent、预测版本、oracle、预算、真实运行证据 |

## 身份、来源和状态

`environment_id`、`workload_id` 和 `trial_id` 应稳定且可追溯。正式记录增加文件校验和；patch 必须能定位到 parent commit。一个 trial 对应一种候选及其多次重复，原始重复测量保留在 `artifacts`，不能只存最快值。

证据状态取 `hypothesis / derived / literature / simulated / user_report / measured / validated / inconclusive`，含义见 [研究方法](../../docs/methodology.md)。候选的 `decision` 独立记录 `pending / accepted / rejected / inconclusive / blocked`，避免把“代码已生成”误认成“性能已验证”。一次失败测量同样是 measured 证据，但其决定可以是 rejected。

严格下界、校准预测和情景预测在 `prediction.kind` 中区分；实际观测独立放在 `measurements` 并登记来源，不放入事前预测。任何区间声明都写明构建方式。`correctness.status=unknown` 不能被当作通过。

## 预算与失败

`planned` 用于决策，`reserved` 用于硬预算控制，`actual` 用于记账。编译失败、超时和正确性失败仍记录实际消耗。GPU、CPU、LLM 和墙钟单位独立，不直接相加。真实控制器对设备租约和超时的处理见 [系统设计](../../optimizer/systems/README.md)。

跨运行统计以独立重复为单位；缺失结果保留原因。在最终报告中同时链接环境、工作负载、候选、原始证据和统计程序。公共记录需移除令牌、内部地址和私密输入，保留足够的可复现定义与 hash。

现有 CPU 原型的 `status=simulated` 对应通用记录的 `evidence_status=simulated`，`evidence_origin=synthetic` 表示输入来源；它们不改变候选的 accepted/rejected 决策语义。
