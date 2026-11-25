# 本轮理论更新记录（Agent 填写）

- 项目 / round / parent / candidate / context:
- active theory 版本与引用章节:
- 预测冻结 ID / comparison receipt:
- 证据类别、正确性、可比性、已知限制:

## 三层对比与偏差
| 层 | 理论预计 | 实际观测 | 支持 / 否定 / 未辨识 | 原始证据 |
| --- | --- | --- | --- | --- |
| 资源 | | | | |
| 执行 | | | | |
| 服务 | | | | |

## 解释与竞争假设
区分事实、推断及其他解释。不得用事后修订覆盖冻结预测。

## 理论更新决定
keep / calibration / correction / extension / empirical / revision；即使 keep 也写明原因。
说明参数或公式变化、域、物理约束、反例、训练证据、候选版本和留出计划。
若仅有一次或不可比证据则保留为 hypothesis，不声称已晋升。

## 下一轮
最小区分办法、预算、候选排序变化、停止/重启条件。

## v2 模块级复盘

将 result/module-review-template.json 复制成轮次根目录 theory-update.json。为所有 primary model/metric 组合填写 decision 和 rationale；必要时登记先修模块。保留 model_gap；记录候选修订 ID、revision、拟合 context、数据引用、下游影响、需要的独立验证。模型引用只表示预测出处，不等于误差因果定位。
