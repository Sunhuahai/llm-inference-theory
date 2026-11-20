# 第二轮模型审计 · Qwen3.8-Flash-Next-FP8 / H800

日期：2026-11-20（Asia/Shanghai）。研究仓库基点：`eab2d2fdce3f22ee596c9b11ce6e50845fa16de8`。
上游 vLLM 源码固定为 `12924016eb8dbaae87d5163a1b7ec2614a44a797`。本文核对上游实现；用户本机引擎、SHA、H800 数量与实际配置仍以部署记录为准。

深入任务卡另核至 `aefea5c8e00c47032cda6d0b75704f6cd0a7c66d`，见 [R01–R03](tasks/README.md)。PLE 的 `ple_layer_ids=[2]` 是 1-based 配置，实际 Python 模块为 `language_model.layers.1.ple`；执行时不要把它解释为索引 2。

**第二轮按具体模型执行路径选题。** 用户已反馈首轮全部尝试，不回到 T001–T013 从头扫开关。源码中的已有能力只作去重依据，不能据此认定本机已经开启。

## 1. 需要据此选优化的模型结构

来源：[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)、[官方 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json)及上述固定源码。

| 模块 | 该 checkpoint 的结构 | 对 H800 优化的含义 |
|---|---|---|
| 主体 | 48 层；36 Gated DeltaNet + 12 QSA | 不能按 48 层普通 dense attention 的 KV 模型估收益 |
| GDN | V heads 48、QK heads 16、key/value head dim 128；config 指定 SSM state FP32 | 大 recurrent state 读写和投机状态提交是独立成本 |
| QSA | Q heads 24、KV heads 2、head dim 256；4-token micro-block、budget 2048 tokens | sparse indexer、compressed/raw scratch 和变长元数据都要保留 |
| GR / HC | 4 residual streams，hidden 2560，低秩宽度 320 | 频繁出现 10240↔320 的小矩阵和门控/归约，适合专门研究小 M 执行 |
| Routed MoE | 512 experts，top-10；expert intermediate 640 | 与常见 top-8、宽专家的 benchmark 形状不同；FP8 block 切分须核对 |
| Shared expert | 另有 1 个 shared expert，intermediate 640 | 不能把它和 routed expert 的 dtype/执行路径混为一谈 |
| PLE | 配在 `ple_layer_ids=[2]`；16 个 n-gram hash heads，每行 160；约 51B 参数 | 优先考虑精确局部行复用；整张表 GPU 常驻不是默认可行 |
| LM head | hidden 2560，padded vocab 248320，config 的 FP8 排除列表含 `lm_head` | 大词表输出路径值得单独 profile，但要保留 MTP/采样需要的 logits |
| FP8 | 主权重采用 128×128 block scale，存在大量排除模块 | “模型 FP8”不等于每个算子 FP8，更不等于可以改成 MXFP8 |

官方 config 属检索日滚动证据，不代表本机 checkpoint revision 已登记。所有量化排除以完整路径与加载后的实际 tensor 为准。

## 2. GDN 状态的量级：先算清优化对象

仅计算 36 层 GDN 的核心 recurrent matrix，不计卷积历史、QSA、PLE、padding 和投机附加副本。按标准 head 分片：

$$
M_{\mathrm{state,aggregate}}
=36\times48\times128\times128\times4
=113246208\ \mathrm{bytes}
=108\ \mathrm{MiB/request}.
$$

当 PP=1、真实 TP 为 P、value heads 合法均分且 state 未额外复制时，每 rank 为 `108/P MiB/request`。例如 **TP=8 只是量纲示例**，是 13.5 MiB/rank/request；不是本机卡数假设。有 PP 时每 rank 应按 `n_local_GDN×(48/P)×128×128×4 bytes` 计算。若每步完整读写一遍，流量约为这个 payload 的两倍。缓存层次、kernel fusion、投机布局和实际带宽决定真实 DRAM 流量，不能把此算式直接当测量值。

这就是 [T017](tricks/T017-replayssm-fixed-precision.md) 的模型依据。避免每个候选 token 都复制完整状态，可能比继续优化已经稀疏的 QSA 更有空间，但排序仍由真实关键路径决定。

## 3. 两个足以改变推荐顺序的源码发现

### 3.1 “支持 MTP”不代表该模型支持 Adaptive Verification

通用 [vLLM v0.34.0 文档](https://docs.vllm.ai/en/v0.34.0/features/speculative_decoding/adaptive_verification/)涵盖 MTP；目标模型的后端组合仍有硬门槛。

本次固定 SHA 审计发现：GDN 已声明 device/CPU query lengths 可不一致；PLE 的 short-conv 后端缺该能力，QSA 与其 indexer 还缺变长 CUDA Graph 的 query length 上界。详见 [T015 的符号与检查链](tricks/T015-adaptive-verification.md)。

因此 T015 记为 **当前上游 vLLM 路径 blocked，需要模型适配**，撤下立即可测清单。不能靠跳过能力检查强开，更不能只复制通用开关。其他引擎需要分别核对，不从 vLLM 的结论推断一律不可能。

### 3.2 GR/HC 已有 PDL，不能再把“打开 PDL”算新方案

Qwen4Exp NVIDIA 路径已含依赖同步、低延迟小 GEMM 与若干 HC 融合。真正剩下的模型专属问题是 **小 M 时哪些组合仍回退到分离计算**，以及现有阈值是否适合 H800。

[T016](tricks/T016-hc-small-m-fusion.md)具体研究 HC 上投影和门控融合的小 M 路径、最终 mixer 的剩余分离边；PDL/MegaOps 只作机制依据，不是新开关。不得用普通 Qwen3/Llama 的 BF16 demo 冒充本模型优化。

## 4. PLE 的精确可缓存部分

当前实际顺序为：

`n-gram IDs → 原格式行读取 → 反量化 → kv_proj → hidden-dependent gate → short convolution`。

前四步由权重及精确 row IDs 决定；后两步依赖上下文与卷积历史。[T020](tricks/T020-ple-exact-cache.md)因此只缓存 owner 内的原字节行，或独立探索 `kv_proj` 输出复用。它可以在完整前缀不同的请求间重用相同局部 n-gram，机制不同于 T008 prefix cache。

现有 pinned-host FP8 lookup 的 ETP 归并依赖唯一 owner；本轮不破坏该不变量。普通 Engram 文档中 DeepSeek 的 peer-group 配置在当前代码里不直接支持 Qwen4Exp，也不搬用 DeepSeek 的表格式和 hashing 规则。

## 5. MoE 的具体约束

- `640/8=80` 无法直接满足原 128 block 对齐；当前源码对 TP4 可无损重复 scale 为 32 网格，TP8 则可能在指定 backend 下走 local 80→128 的完整块 padding，总分配宽度为原来的 1.6 倍。支持与效率须看实际 loader/backend；旧 recipe 的限制不能扩展为永久不支持。详见 [T018](tricks/T018-tile-collective-fusion.md)。
- 固定 Qwen4Exp `mega_moe.py` 存在把原 block-FP8 重量化为 MXFP8 的加载路径，且有特定 TP/硬件约束。本轮固定精度，不能把这个后端转换包装成普通通信优化。[T018](tricks/T018-tile-collective-fusion.md)保留原数学 kernel、scale 和 dtype，只研究合法的通信边；[T019](tricks/T019-bounded-ep-workspace.md)只研究缓冲生命周期。
- EP 不自动意味着 DeepEP dispatch/combine；普通 TEP8、DP=PCP=1、无 MoE-SP 的 Triton 路径可能由输入副本、本地专家和 late all-reduce 完成。先核 prepare/finalize class。non-expand 接收按目标 rank 去重，而专家计算仍保留 top-10；不能把通信量直接乘十。
- 对本模型 standard Triton 路径，两块专家 workspace 都是 `[R,10,2560]` BF16，共 `102400R` 字节。R=8192 时的 800 MiB 只是源码容量算例；R 是 prepare 后行数，非请求数。已有 bucket/alias 与 Graph 高水位会影响能否真正释放，详见 T019。

## 6. 交给实施端的最小补充证据

从已完成首轮的环境提取即可，不要求重跑首轮：当前最佳 baseline ID/commit、真实 GPU/进程组、各模块实际 dtype、一次 decode profile、HC 实际 M 分布、GDN 写回/回滚耗时、PLE 暴露等待、head+sampler 占比。

任何“已默认含此优化”的结论都应由本机调用路径证明。本文没有用户 H800 实测收益。
