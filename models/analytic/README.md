# 多资源与执行图解析参考

> 状态：可运行的标准库 CPU 参考，2026-11-25。示例为 `SYNTHETIC`，没有 GPU 测量。实现为[资源模型](../resource/README.md)与[执行图](../execution/README.md)的窄子集。

## 1. 运行与合成结果

在仓库根目录：

```bash
python3 models/analytic/analyze.py models/analytic/examples/synthetic-system.json
python3 -m unittest discover -s models/analytic/tests -v
```

可用 `--output result.json` 保存输出。示例包含 Attention → Projection/Transfer 分支，得到固定图的条件时间下界 12、6、5 ms：减少流量后 Attention 从 HBM 限制迁移到计算限制；进一步减少传输后，关键路径迁移到 Projection。它们是对合成输入的计算结果，不是模型/硬件性能成绩。

## 2. 输入 schema

| 字段 | 类型与语义 |
| --- | --- |
| `schema_version` | 整数 1 |
| `input_origin` | `synthetic` / `literature` / `user_report` / `measured`；来源由输入方声明 |
| `mode` | `bound` 或 `forecast` |
| `task_signature` | 所有候选相同任务、精度和资源预算的非空签名；程序不能证明语义相同 |
| `baseline_id` | `scenarios` 中的一个 ID |
| `resources` | 按资源 ID 定义 `rate`、`work_unit`、`rate_unit`、`rate_kind`、`source` |
| `scenarios` | 非空列表；每项有唯一 ID、图、内存输入 |
| `observations` | 可选观测列表，仅作对照；详见[统一契约](../contracts.md) |

`work_unit` 只接受 `FLOP`、`byte`、`operation`；`rate_unit` 必须为对应单位加 `/s`。输入数值必须有限、非负，速率和可用容量严格为正。物理资源需按设备/链路 ID 分开，不能把两个独立 GPU 的算力当同一速率再重复使用。

`bound` 要求全部速率类型为 `physical_upper`，表明输入方声称它们是适用范围内的速率上界；`forecast` 要求 `effective_forecast`。程序拒绝两者混用，但不会自动验证硬件规格或校准证据。实际工作量高于输入的理想必要工作，会使边界更松，不会让边界自动变成预测。

每个场景具有：

- `graph_semantics`：`dependencies_only` 或 `fixed_schedule`。
- `nodes`：每个节点有唯一 `id`、`after` 前驱 ID 列表、`demands`。
- `demands`：资源 ID → `{amount, unit}`，同节点不同资源时间按 `max` 组合。
- `serialization_orders`：固定顺序模式的节点序列列表，将相邻节点加入先后边。
- `memory`：`required_peak_bytes`、`available_bytes`、`source`。峰值由外部提供，程序不推导内存生命周期。

如果两个节点共享资源且没有先后关系，`fixed_schedule` 会拒绝。使用 `dependencies_only` 则保留并行可能性，下界模式同时考虑全图总资源需求；它并不自动安排并发。

## 3. 输出与能力边界

输出包含逐资源时间、节点限制资源、最长路径、全图资源总时间、容量可行性及 `value_kind`。资源下界和图下界取最大值；固定排程模式的下界只约束该排程。预测模式使用有效速率返回图估计，不宣称严格下界。

基线/候选比值命名为 `bound_ratio_not_speedup` 或 `model_time_ratio_not_measured_speedup`。不可行场景或不同图语义间不生成比值。相同资源预算和任务的真实性由外部 manifest/评估器检查。

当前不支持任意 concurrent kernels 的资源份额、通信拓扑自动推导、显存生命周期、队列、动态 batch、冷启动、TTFT/TPOT/p99 或自动收益学习。新增这些能力需要相应数据与验证，不能通过给输出改名完成。

## 4. 最小验证覆盖

测试覆盖共享资源的错误免费并发、总资源下界超过最长路、瓶颈与关键路径迁移、单位/非有限数/依赖环错误、有效速率误标为物理上界、容量不可行，以及观测证据来源和留出标签的保留。运行结果证明这些计算约束得到执行；实测可预测性仍需[模型验证](../../experiments/model-validation/README.md)。
