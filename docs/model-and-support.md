# 模型身份、支持边界与实施门槛

初次检索：2026-11-16；第二轮源码更新：2026-11-20。此处记录上游信息；用户部署环境见 [environment.yaml](../environment.yaml)，当前尚未登记。用户已反馈首轮尝试完成，不能将环境字段缺失解释为尚未部署。

**当前研究入口：**[Qwen4Exp 第二轮源码审计](../weeks/2026-W47/model-audit-round-02.md)、[实施任务书](../weeks/2026-W47/agent-tasks-round-02.md)。本轮固定审计 vLLM `12924016eb8dbaae87d5163a1b7ec2614a44a797`，以 HC 小 M、GDN state 和 PLE 精确缓存为主线。

## 一、目标模型

用户所称“千问 3.8 Next 8 比特”与官方 [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) 匹配。本机路径可能是副本或另一种 8-bit 制品，执行前必须核对完整 ID、revision 和 config；不得静默转换权重。

| 核对项 | 官方信息 | 对优化的含义 |
|---|---|---|
| 架构标识 | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` | 不能按旧 Qwen3-Next 的模块与开关操作 |
| 混合注意力 | 48 层，Gated DeltaNet 与 QSA 混合 | 同时考虑 recurrent state、稀疏索引与 KV |
| MoE / GR / PLE / MTP | 稀疏专家、Gated Residual、N-gram embedding、原生 MTP | 算子、CPU 取表、通信与推测解码都可能是瓶颈 |
| 参数统计 | 125B 主体 + 51B PLE；另有约 4B MTP | 总量/激活量不能直接换算每卡运行显存 |
| FP8 制品 | 128×128 block，保留部分不同 dtype 模块 | 8-bit 是固定输入条件，不是本项目优化变量 |

依据：[Qwen 官方说明](https://github.com/QwenLM/Qwen3.8-Flash-Next)、[官方模型 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json)。上表只提取影响实验选择的属性，不复述完整模型卡。

## 二、框架门槛

[vLLM 官方 recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml) 初轮记录的最低版本为 `0.30.0`，调优示例使用 nightly；[SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx) 初轮记录为 `>=0.5.22`。它们是参考门槛，不证明用户安装包已具备某个优化。执行时记录实际 release、git SHA 或镜像 digest，检查实现与 resolved 配置。

特别注意 vLLM recipe 对官方 FP8 **plain TP8 的 block 分片限制**，参考八卡方案使用 TEP8。11 月 20 日源码另有 scale refine 与完整 block padding 分支：TP4 的 160 宽可以无损重复 scale，TP8 的 80 宽在某些显式 backend 下可能补至 128，但总分配宽度增加为 1.6 倍。实际支持与效率依赖 loader、backend 和真实进程组，详见 [T018](../weeks/2026-W47/tricks/T018-tile-collective-fusion.md)；不能把旧 recipe 限制解释为所有新版本永久不支持。Hopper 的 MoE backend、PLE offload 和模型并行逐项核对，不能仅凭“8 卡”指定布局。

用户 H800 的 SKU、卡数、显存、NVLink/PCIe/NUMA/网络都未知。H100、H200、B300 结果仅用于提出候选，不作为本仓库 H800 性能证据。

## 三、能力检查清单（第二轮复用已有记录）

以下字段若已在首轮记录，直接复用；只补当前缺失项，不要求重跑首轮：

| 能力 | 要提供的证据 | 未满足时 |
|---|---|---|
| checkpoint 身份与所有 dtype 固定 | config 摘要、revision、实际加载日志 | 停止可比性能实验，记录 unknown |
| H800 与进程组拓扑 | 脱敏 `nvidia-smi`、拓扑、NUMA 与各 rank 资源 | 只保留适用的候选，不假设多卡 |
| 模型加载与基本输出正确 | 最小请求、输出终止与错误检查 | 先解决部署问题，不能把失败作为低延迟 |
| 当前融合路径 | 源码/启动日志/少量 profile | 已包含则标 baseline-covered |
| CUDA Graph / SP / MTP / prefix state 支持 | 当前版本帮助、源码与模型测试 | 不支持则给证据，不猜开关 |
| 显存与 CPU 表容量 | 各 rank 峰值、host RAM、PLE offload 状态 | 先限定合法布局和负载 |

## 四、固定精度不等于禁用高效内核

允许选择能够消费**同一 checkpoint 和同一 dtype**的更快实现、改变数据布局、调度或融合。禁止把权重/KV/indexer/SSM/PLE 改成更低精度后称为调度优化。原有 `auto` 也要记下实际解析结果，避免升级框架后默认值悄悄改变。

官方启动示例中的 cache dtype、thinking 设置、最大输出与硬件配置必须逐项审查；示例不是本机可以原样执行的实验命令。每条卡片的 A/B 方案都需要通过这里的门槛。

## 五、第二轮特有的支持边界

- 目标当前源码位于 `vllm/models/qwen4_exp/{common,nvidia}`，不同于旧 Qwen3-Next。HC 已含 PDL 和部分融合，新增点是小 M 分派与剩余融合边。
- GDN 核心 recurrent payload 按官方 FP32 配置为全模型 108 MiB/request；本地容量按 TP/PP 和实际 tensor 算，不含其他状态。ReplaySSM 的已有 Qwen4Exp PR 走 SM100/FP16，H800 固定精度版本需要移植。
- Adaptive Verification 的通用 MTP 支持不等于这个模型当前可用；所审计版本的 PLE、QSA/indexer 有能力检查阻塞。
- PLE 可以研究静态行/投影复用，但 hidden-dependent gate 和 short-conv 不属于静态缓存对象；必须保留 ETP 唯一 owner。
- 第二轮方案均未在用户 H800 上由研究 Agent 实测。对应版本、检查链和限制见[本轮来源](../weeks/2026-W47/sources-round-02.md)。
