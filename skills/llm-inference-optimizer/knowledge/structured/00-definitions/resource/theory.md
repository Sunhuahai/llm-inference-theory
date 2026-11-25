# 资源与量纲

模型 ID：`DEF-RESOURCE` · revision 1 · `definition` · [机器可读定义](model.json)

## 核心关系

`work / rate = time; peak capacity != effective rate`

FLOP、byte、s 为核心 SI 量；容量和流量不同，明确每卡/全局。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| resource_spec | record | 设备、工作量、速率与容量的统一账本 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

有效速率是依工作点变化的估计，不能替代严格上界。

## 验证与反例

检查单位换算、重复记账、rank/链路身份与规格来源。

## 依赖与来源

依赖 ID：无。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [models/contracts.md](../../../initial/models/contracts.md)
- [models/resource/README.md](../../../initial/models/resource/README.md)

本模块为理论/模型规范，暂无通用可执行 evaluator。
