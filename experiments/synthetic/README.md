# 合成验证与 CPU 示例

合成数据用于检验推导与实现是否一致、发现反例以及验证控制逻辑。它没有提供真实 GPU 的效率、缓存命中分布或 Agent 能力证据。

| 示例 | 研究对象 | 入口 |
| --- | --- | --- |
| 解析资源与执行图 | 资源边界、串并行、瓶颈迁移 | [Analytic](../../models/analytic/README.md) |
| 已有收益场景 | 基线归一化、MTP、资源需求与缓存 trace | [Predictor](../../models/predictor/README.md) |
| 离线搜索回放 | 预算、候选选择、正确性与 incumbent | [Search replay](../../optimizer/systems/README.md) |

从仓库根目录运行：

```bash
python3 models/analytic/analyze.py models/analytic/examples/synthetic-system.json
python3 models/predictor/predict.py models/predictor/examples/synthetic-baseline.json --format json
python3 models/predictor/cache_context.py models/predictor/examples/synthetic-cache-context.json --format json
python3 optimizer/systems/replay.py optimizer/systems/examples/synthetic-search.json --policy cost-aware --budget 12 --seed 0
```

所有输入值由示例构造，输出继续标为 `simulated`/`SYNTHETIC`。修改预测器、候选表或随机种子可以改变结果；不能用这几条示例评选最好的搜索方法。

最低检查包括零改动还原、预算零/耗尽、环或缺失依赖、单位与非法数值、正确性失败、预测方向错误，以及两项优化组合后出现瓶颈迁移。测试记录见各实现 README；真实外部验证要求见 [模型验证](../model-validation/README.md) 与 [Agent benchmark](../agent-benchmark/README.md)。
