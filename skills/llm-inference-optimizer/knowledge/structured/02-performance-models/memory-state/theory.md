# 状态容量、流量与复用

模型 ID：`PERF-STATE` · revision 1 · `performance_model` · [机器可读定义](model.json)

## 核心关系

`M_KV=2*L*B*S*H_kv*d_head*bytes_per_element (uniform, unsharded)`

均匀 KV 形状、未分片且不计 padding；混合架构按层独立累加。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| layers | count | 层数 |
| batch | count | 序列数 |
| length | count | 每序列 token 数 |
| kv_heads | count | KV head 数 |
| head_dim | count | head 维数 |
| element_bytes | byte | 元素字节数 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| state | byte | 理想未分片 KV 大小 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

实际驻留另含 allocator、页、索引、workspace；缓存复用有恢复和容量代价。

## 验证与反例

比较实测峰值、访问计数与 cache trace；检查分片/复制与驱逐。

## 依赖与来源

依赖 ID：`PERF-RESOURCE`、`FND-IO`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/memory/README.md](../../../initial/theory/memory/README.md)
- [theory/foundations/cache-context-ceilings.md](../../../initial/theory/foundations/cache-context-ceilings.md)

具备窄解析参考函数：kv_capacity；输入必须通过单位和适用域检查，结果不是实测。
