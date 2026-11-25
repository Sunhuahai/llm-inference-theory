# 多资源 Roofline 下界

模型 ID：`FND-ROOFLINE` · revision 1 · `physical_bound` · [机器可读定义](model.json)

## 核心关系

`T >= max(F/P_max, D/B_max)`

针对同一区间的必要工作量、真实速率上界；max 是必要条件而非可达调度。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| flops | FLOP | 必要计算量 |
| bytes | byte | 必要流量 |
| peak_flops | FLOP/s | 计算速率上界 |
| peak_bandwidth | byte/s | 带宽上界 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| time | s | 时间下界 |
| intensity | FLOP/byte | 算术强度（D>0） |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

未建模启动、通信、依赖、资源争用；使用有效速率时仅是情景估计。

## 验证与反例

用微基准核查输入上界与工作量；实际时间低于声称下界说明前提/数据有误。

## 依赖与来源

依赖 ID：`FND-COMPUTE`、`FND-IO`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/performance/README.md](../../../initial/theory/performance/README.md)
- [theory/foundations/unified-theory.md](../../../initial/theory/foundations/unified-theory.md)

具备窄解析参考函数：roofline；输入必须通过单位和适用域检查，结果不是实测。
