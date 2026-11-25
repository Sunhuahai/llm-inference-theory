# 服务级性能与 Goodput

模型 ID：`PERF-SERVING` · revision 1 · `performance_model` · [机器可读定义](model.json)

## 核心关系

`Y=Simulate(arrivals, scheduler, execution, state, generation); goodput=valid_SLO_completions/window`

固定 trace、资源、质量、计时和指标定义；动态状态需模拟或实测。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| workload | record | 到达、长度与 SLO 联合分布 |
| scheduler | record | 批处理、路由和资源分配 |
| execution | record | step 时间、容量及有效生成 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| ttft | ms | 请求 TTFT（统计量另定） |
| tpot | ms/token | 请求 TPOT（统计量另定） |
| throughput | tokens/s | 提交输出吞吐 |
| goodput | requests/s | 满足定义的有效请求率 |
| p99 | ms | 已明确时延指标的尾分位 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

没有通用服务模拟器；低负载或饱和数字不代表混合到达下 p99。

## 验证与反例

不同到达和长度联合分布的未见 trace 对比；校验丢弃/完成及 SLO。

## 依赖与来源

依赖 ID：`PERF-EXECUTION`、`PERF-GENERATION`、`PERF-STATE`、`FND-QUEUEING`、`FND-FEASIBILITY`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/serving/README.md](../../../initial/theory/serving/README.md)
- [models/serving/README.md](../../../initial/models/serving/README.md)
- [models/simulation/README.md](../../../initial/models/simulation/README.md)

本模块为理论/模型规范，暂无通用可执行 evaluator。
