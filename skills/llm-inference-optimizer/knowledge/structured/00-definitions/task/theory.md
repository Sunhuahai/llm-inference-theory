# 任务与不变量

模型 ID：`DEF-TASK` · revision 1 · `definition` · [机器可读定义](model.json)

## 核心关系

`TaskSpec = checkpoint + tokenizer + precision + semantics + quality contract`

候选的比较必须保持任务语义和质量约束；可调整变量由项目声明。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| task_spec | record | 任务语义签名 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

不能从速度推断质量。任务变化必须单列研究。

## 验证与反例

对照固定 tokenizer、采样、功能、质量 oracle 和资源预算；记录实际生效后端。

## 依赖与来源

依赖 ID：无。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/foundations/unified-theory.md](../../../initial/theory/foundations/unified-theory.md)
- [models/contracts.md](../../../initial/models/contracts.md)
- [docs/benchmark-protocol.md](../../../initial/docs/benchmark-protocol.md)

本模块为理论/模型规范，暂无通用可执行 evaluator。
