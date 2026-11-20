# T020 · Qwen4Exp PLE 的精确行缓存与静态投影复用

- 状态：`conditional`；本轮主线候选。证据来源：`literature + hypothesis`。
- 调研日期：2026-11-20（Asia/Shanghai）。本机结果：`unknown`。
- 目标：现有 Qwen3.8-Flash-Next-FP8 / H800；沿用 Week 1 最佳有效配置与全部实际 dtype。
- 新增点：在已做过的 T006 异步预取之上，减少重复查表本身；与 T008 的完整前缀/状态缓存不同。
- 这是本项目根据模型源码提出的实现实验，不是已经公开验证的 Qwen3.8/H800 加速功能。

## 1. 为什么只对这个模型提出

[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json)给出 `ple_layer_ids=[2]`、`ngram_size=3`、`heads_per_ngram=8`、`ple_embed_dim=2560`。固定 [vLLM 12924016](https://github.com/vllm-project/vllm/tree/12924016eb8dbaae87d5163a1b7ec2614a44a797) 的实现得到：

| 结构 | 目标模型中的具体值或行为 |
|---|---|
| 查表头数 | (3−1)×8 = 16；二元和三元 n-gram 各 8 个头 |
| 每行宽度 | 2560/16 = 160 个元素 |
| 单 token 原始读量 | 16×160×b = 2560b 字节，b 为实际表元素字节数；若原表为 FP8，主 payload 为 2560 字节 |
| ETP 所有权 | 每个全局 row ID 恰有一个 owner；其他 rank 写零，再 reduce 或 reduce-scatter |
| 表的原格式 | `Qwen4ExpPLEFp8EmbeddingMethod` 路径使用 FP8 原字节和原全局 scale；与专家 128×128 block scale 不同 |
| PLE 投影 | 2560 → (4×2560 + 2560) = 12800，`kv_proj` 的权重保持 replicated |
| 已有预取 | 在前一层发起 side-stream UVA lookup；不能把再次开启它算新实验 |

上表是官方配置与源码推导，不是用户部署确认。实际表格式、ETP 和 HC-SP 均读本机快照。

## 2. 准确落点

所有路径均相对上述固定 SHA：

- [common/ngram_embedding.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/common/ngram_embedding.py)：`Qwen4ExpPLEPinnedHostEmbedding._lookup`、`Qwen4ExpPLEEmbeddingMethod.lookup_from_pinned`、`_lookup_ple_embedding_from_pinned_kernel`、`_reduce_etp_embeddings`。
- [nvidia/ngram_embedding.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/ngram_embedding.py)：`compute_ngram_ids`、`start_prefetch`，以及 ETP/HC-SP 的输出归并。
- [nvidia/ple_layer.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/ple_layer.py)：`forward` 中的 `embedding → dequantize → kv_proj → ple_gate → short_conv`。
- [nvidia/model.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/model.py)：`_start_layer_ple_prefetch`。
- [config/engram.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/config/engram.py)：当前没有本方案的 hot-row cache 配置；`peer_group_size` 的现成路径限制为 DeepSeek V4.1，不是 Qwen4Exp 的可用替代。

当前 pinned-host kernel 对每个 row 请求直接从 UVA 地址读取；没有在所审计路径发现软件管理的 GPU 热行缓存。GPU 硬件缓存是否已消除重复传输仍须测量。

## 3. 第一版：owner 内的只读热行缓存

先用已有脱敏运行记录采集全局 row ID 频率和时序，再用独立负载评价；不能只在选择热行的同一段 trace 上报告命中。

在每个 owner rank 放一小块固定容量 HBM cache，缓存 checkpoint 原始行字节；miss 沿用原 UVA lookup。键含模型/表/权重版本与完整全局 row ID，不能把有碰撞的短 hash 当相等证明。原 FP8 全局 scale 与反量化次序不变。初版离线选热行，启动后只读，避免每步插入、驱逐和锁竞争。

**只允许 owner 提供非零数据。** 当前 FP8 ETP 通过 int8 视图归并原始字节，依赖唯一 owner；若把同一行在多个 rank 同时作为有效命中相加，会直接破坏数据。缓存不改变这一不变量，也不顺带更换 collectives。

第一轮只改变 lookup 数据来源，保持输出 shape、预取时机、请求 batch、HC-SP、ETP 和原缓冲结构。预算取实际可用显存中的 2–3 个小档位；同时报告占用的 HBM 是否挤压 KV/GDN state 并发容量。

## 4. 更激进但独立的第二版：缓存投影结果

源码中有一段完全不依赖当前 hidden state 的静态计算：

$$
e_t=E(\mathrm{ngramIDs}_t),\qquad [k_t,v_t]=W_{\mathrm{PLE}}e_t.
$$

可用完整 16 个 row IDs 和权重/精度版本作键，缓存现有 `kv_proj` 产生的 `[k,v]`。命中可跳过查表、反量化和这次投影；**不能缓存后续 gate、hidden residual 或短卷积输出**，它们依赖当前上下文和卷积历史。

若输出为 BF16，单条 `[k,v]` 主 payload 为 12800×2 = 25600 字节，约 25 KiB，显著大于单条原始行。此处只是容量计算，不是假定命中率。

这一版单独与 A 对照，不能与热行缓存同时上线后归因。当前 UVA lookup 已在前一层 `start_prefetch` 发起；若只在 `ple_layer.forward` 查询投影 cache，无法省掉已发出的查表。要省 lookup，probe/miss-mask 必须前移到 prefetch 之前，同时保持 ETP/HC-SP collective 的参与和 shape；否则只计实际省下的反量化/投影。

miss compaction 可能把 GEMM 变小；CUDA Graph 下逻辑屏蔽命中行若仍执行完整 GEMM，不会节省投影计算。需证明实际少算，并对批形状改变导致的浮点差异作同精度核验。

## 5. 最小实验与退出条件

| 对照 | 唯一变量 | 要排除的问题 |
|---|---|---|
| A ↔ B | 原 lookup / owner 内只读热行缓存 | 软件查键成本是否超过省下的 UVA 等待 |
| A ↔ C，B 完成后独立做 | 原静态路径 / 精确 `[k,v]` 复用 | 节省投影是否被 miss 小 GEMM、索引与 gather 抵消 |
| B 的冷 ↔ 暖分层 | 相同容量、固定状态准备方式 | 命中只是重复完整 prompt，还是跨请求局部 n-gram 也有效 |

主指标仍是 TTFT/TPOT/输出吞吐与失败率。机制指标增加 row hit、唯一 row 数、真实 PCIe/host bytes、查键耗时、暴露的 prefetch 等待、miss GEMM 形状、cache MiB、可运行 state slot 数。交错至少三次；不用 warm-only 数字代替真实 trace。

本项目的粗略盈亏判据为：

$$
h\,T_{\mathrm{avoided,exposed}}>
T_{\mathrm{probe}}+T_{\mathrm{fill}}+T_{\mathrm{contention}}+T_{\mathrm{capacity\ loss}}.
$$

各项需来自同一 critical path；已完全被前层隐藏的 lookup 不能再算节省。冷场景尾延迟、命中低、容量减少或查键成本过高时退出，恢复原 lookup 和原 pool 大小。

正确性覆盖真实 n-gram 边界/EOS、padding、重复 IDs、跨 owner、HC-SP、DP、MTP 接受与拒绝、权重 reload。原始行命中应逐字节一致；投影复用另比较固定隐藏态下的 logits/任务质量。权重更新必须失效所有旧表条目；不能用近似语义匹配替代精确 key。

结果：`baseline_id=unknown`，`candidate_id=unknown`，`H800_gain=unknown`。
