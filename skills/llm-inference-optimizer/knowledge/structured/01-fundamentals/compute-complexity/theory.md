# 计算工作量与复杂度

模型 ID：`FND-COMPUTE` · revision 1 · `derived` · [机器可读定义](model.json)

## 核心关系

`F_GEMM=2*M*N*K (multiply-add counted as 2 FLOP)`

密集 GEMM 的乘加计数；明确是否包含 epilogue、稀疏和重算。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| m | count | 行维度 |
| n | count | 输出列 |
| k | count | 归约维度 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| flops | FLOP | 逻辑乘加工作量 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

不等于实际 issued instructions；padding、专家路由、特殊算子需另计。

## 验证与反例

对照 shape、算子图、真实指令或 profiler；检查计数口径。

## 依赖与来源

依赖 ID：`DEF-RESOURCE`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/foundations/unified-theory.md](../../../initial/theory/foundations/unified-theory.md)
- [models/resource/README.md](../../../initial/models/resource/README.md)

具备窄解析参考函数：gemm_flops；输入必须通过单位和适用域检查，结果不是实测。
