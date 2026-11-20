# 模型与精度合同

## 1. 先识别真实环境

研究对象是 `Qwen/Qwen3.8-Flash-Next-FP8`，架构 `Qwen4Exp`，目标硬件 H800。实际 checkpoint/revision、GPU 数量、引擎和工作负载以运行记录为准。读取 [environment.yaml](../../../environment.yaml)；缺字段保持 unknown，不套官方示例。

新版 vLLM 的相关路径是 `vllm/models/qwen4_exp/{common,nvidia}`。Round 03 文档的统一参考 SHA 是 `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`；这是研究源码基点，不是要求用户升级。其他引擎按实际等价实现核对，不能直接照搬开关。

## 2. 固定合同

| 对象 | 必须记录并保持 |
|---|---|
| 身份 | checkpoint/revision、tokenizer、chat template、实际架构 |
| 权重 | 原有 dtype、量化格式、原始 block scale 与 loader 变换；不能重铸 FP8 后只比较 dtype 名称 |
| 计算 | 激活与计算 dtype、累加精度、TF32 设置、quant/dequant 的 cast 顺序、归约语义 |
| 状态 | KV、indexer cache、GDN/SSM、PLE 的实际 dtype 与生命周期 |
| 任务 | 相同 prompt/trace、采样、thinking、stop/EOS、输出上限及业务功能 |
| 资源 | 相同 H800 总预算、拓扑、并行组与环境限制；内存收益要报告 host/HBM 代价 |

数据布局、调度或访存顺序可以作为变量；它们不能暗中改变上述合同。保持数学模式不意味着所有合法重排都必须 bitwise 相同：按任务卡预先规定浮点容差；离散索引、mask、路由集合、状态归属与接受路径必须满足对应精确合同。不要看完结果再放宽容差。

## 3. 模型专属陷阱

- GDN、QSA、PLE 共存。能恢复 GDN state 不代表 QSA KV、压缩/indexer 与 PLE n-gram/conv 历史也恢复正确。
- PLE 的静态查表/投影与依赖 hidden 的 gate、short-conv 不同。不能把动态部分放进静态 memo。
- QSA 的索引集合决定实际 attention。要检查 mask、压缩位置、重复/无效索引及 tie 规则；不能把少看一些 token 当作纯内核优化。
- Native MTP 的目标与草稿状态必须隔离；提交已接受 token 时同步所有状态和游标。历史复制 proposal 不能沿用错误的 MTP proposal 概率。
- Routed experts 与 shared expert 不同。保持 top-10 权重、normalization、共享分支与最终 reduction 的归属；不能漏 expert 或把共享分支重复求和。
- 现成后端可能把 block-FP8 转成另一种格式，或者合并 scale 后重新 cast FP8。标签仍叫 FP8 也可能违反合同。
- 官方 `FP8` 制品包含 BF16 等排除模块。路由 GEMM 输出 dtype、softmax 累加 dtype、LM head dtype 分开记录，不能按模块名称推断。

以上源码依据与完整版本链接见 [模型审计](../../../weeks/2026-W47/model-audit-round-02.md)、[Round 03 来源](../../../rounds/round-03/sources.md) 和各任务卡。此页是执行门槛，不是用户机器的实测环境。
