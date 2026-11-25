# 硬件能力与有效速率

模型 ID：`PERF-HARDWARE` · revision 1 · `performance_model` · [机器可读定义](model.json)

## 核心关系

`P_eff=eta_P*P_peak; B_eff=eta_B*B_peak`

设备、频率、dtype、shape、后端、并发与热状态明确。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| hardware_spec | record | 设备、dtype、拓扑和峰值来源 |
| shape | record | 微基准工作点 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| effective_compute | FLOP/s | 条件有效计算率 |
| effective_bandwidth | byte/s | 条件有效带宽 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| eta_compute | ratio | 计算效率；(0,1] 且峰值口径一致 |
| eta_bandwidth | ratio | 带宽效率；(0,1] 且峰值口径一致 |

默认无实测参数集；只能在匹配 context 与证据下拟合，候选须独立验证。

## 适用边界

效率非硬件常数；未知值不默认 1；峰值与持续速率区别。

## 验证与反例

独立微基准拟合后对未见 shape/负载验证，报告不确定性。

## 依赖与来源

依赖 ID：`DEF-RESOURCE`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [models/resource/README.md](../../../initial/models/resource/README.md)
- [models/operator/README.md](../../../initial/models/operator/README.md)
- [models/calibration/README.md](../../../initial/models/calibration/README.md)

本模块为理论/模型规范，暂无通用可执行 evaluator。
