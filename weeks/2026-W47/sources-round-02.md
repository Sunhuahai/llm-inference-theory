# 第二轮一手来源与版本锁定

检索日期：2026-11-20（Asia/Shanghai）。对应[模型审计](model-audit-round-02.md)、[研究综述](research-round-02.md)和 T014–T020。只有一手论文、作者仓库与框架源码作为技术依据；外部性能不记作本机结果。

## 1. 目标模型与共同源码基点

| 来源 | 固定点或范围 | 用途与边界 |
|---|---|---|
| [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) / [config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json) | 检索日的 main；本机 revision 尚未登记 | Qwen4Exp、48 层、36 GDN/12 QSA、HC=4、H=2560、rank=320、512/top10/640、PLE 和 dtype 排除。不能代替实际加载快照 |
| [vLLM 主代码](https://github.com/vllm-project/vllm/tree/12924016eb8dbaae87d5163a1b7ec2614a44a797) | `12924016eb8dbaae87d5163a1b7ec2614a44a797` | 本轮全部目标路径静态审计基点，2026-11-20 读取；不是用户部署 SHA |
| [目标模型实现目录](https://github.com/vllm-project/vllm/tree/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp) | 同上 | 当前路径为 `vllm/models/qwen4_exp/{common,nvidia}`；不要继续从旧目录名推断是否实现 |
| [vLLM recipes](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml) | 滚动文档，检索日 2026-11-20 | 参考部署与旧 plain TP8 限制；当前 refine/padding 行为必须结合固定源码核对 |

模型配置信息和源码事实不等于支持本机版本；所有立即任务都以 Week 1 当前最佳有效配置为比较锚点。其他引擎需映射实际模块，不为匹配本报告而直接换引擎。

## 2. 三个优先方向的直接证据

### T016：HC 小 M 融合

- [hyperconnection.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/hyperconnection.py)：`GatedResidual`、up/down 权重、普通与 final mixer 的差异。
- [ops/hc.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/ops/hc.py)：`MIN_FUSED_UP_M=512`、已有 SM90 up+gate_mix、BF16 gate rounding 与 PDL。
- [hc_down_silu.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py)及[low_latency_gemm.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py)：已有 down 融合、预取及 H200 测得的 skinny 分派，作为去重依据。
- [CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)与[CUTLASS 4.2.1](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/dependent_kernel_launch.html)：依赖同步机制；不保证发生重叠。CUTLASS tag 对应 `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`；示例的 FP8 fast accumulation 不是本模型精度替代方案。

**证据结论：**小 M 分离路径是固定源码事实；H800 上改白名单或新小 M kernel 是否更快是本项目假设。

### T017：GDN ReplaySSM 固定精度移植

| 来源 | 固定点 | 可用证据 / 不可外推的部分 |
|---|---|---|
| [ReplaySSM 论文](https://arxiv.org/html/2610.13778v1) | arXiv v1；特别看 Appendix B.4 | checkpoint+replay 机制；FP32-state 数字仍含 FP16 cached corrections，不能直接借用 |
| [作者说明](https://dao-lab.ai/blog/2026/replayssm/) / [作者代码](https://github.com/Johnny-Liou/ReplaySSM/tree/73a9592e9d8a2268620e8883a82f5cec7442ed36) | `73a9592e9d8a2268620e8883a82f5cec7442ed36` | 单层算法与校验入口；不是用户完整模型的服务实现 |
| [vLLM #62004](https://github.com/vllm-project/vllm/pull/62004) | 检索时 open，head `0880cb87383330070f6a2546536ee99496306bab` | Triton GDN 普通 decode，H100/Qwen3.5 验证，主要针对 serving batch；仍需检查 ring 精度，且不含完整 Qwen4Exp 投机接线 |
| [vLLM #65657](https://github.com/vllm-project/vllm/pull/65657) | 检索时 open，head `682a577e7e50bfcc9786544152244ece7f2c7be2` | 确有 Qwen3.8 接入，但要求 SM100、FlashInfer 0.7.3 和 FP16 state；不能当 H800/FP32 支持 |
| [vLLM #55688](https://github.com/vllm-project/vllm/pull/55688) | merge `750cad1973f037fe00116cbf58cd3b3bd8701cd4` | 共享 Mamba 生命周期已合并，不能据此宣称目标 GDN kernel 已可用 |

目标代码入口：[qwen_gdn_linear_attn.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)、[mamba_utils.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/mamba/mamba_utils.py)、[MambaBase](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/mamba/abstract.py)。108 MiB 是目标 config 的容量推导，不是测量。

### T020：PLE 精确缓存

- [common/ngram_embedding.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/common/ngram_embedding.py)：原格式 UVA 行读取、ETP 唯一 owner、FP8 字节归并。
- [nvidia/ngram_embedding.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/ngram_embedding.py)与[ple_layer.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/ple_layer.py)：16 个精确 row IDs、预取、反量化、静态 kv_proj 与动态 gate/conv 边界。
- [config/engram.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/config/engram.py)：peer group 的现成支持限制，不能把 DeepSeek-only 配置当 Qwen4Exp 开关。

**证据结论：**被缓存对象的静态依赖来自源码；owner 热行与投影缓存是本项目提出的实验设计，尚无本模型/H800 已验证实现或速度数字。

## 3. 储备方向的依据与否决条件

| 方向 | 一手来源与固定版本 | 本轮用途与限制 |
|---|---|---|
| T014 FlashSampling | [论文 v4](https://arxiv.org/html/2603.15854v4)、[代码](https://github.com/FlashSampling/FlashSampling/tree/834fab55b638a32e13e0a0bc525ee5aeebfb0cc9)；作者 [vLLM fork PR #13](https://github.com/tomasruizt/vllm/pull/13)，head `7a74973e4dc727df979f2a5ec9fff64ac5319467` | 原理可借鉴；论文新 top-k/top-p 范围与公开代码不同，且现有 MTP target 验证接口接收 logits。原本纯 greedy 可另研究接口适配；不是上游 main 的现成开关 |
| T015 adaptive verification | [vLLM v0.34.0 文档](https://docs.vllm.ai/en/v0.34.0/features/speculative_decoding/adaptive_verification/)及[卡片的完整能力检查链](tricks/T015-adaptive-verification.md) | GDN 已支持相关变长能力；PLE、QSA/indexer 尚有阻塞，记 blocked |
| T018 细粒度通信 | [FLUX](https://arxiv.org/abs/2406.06858)、[COMET](https://arxiv.org/abs/2502.19811)、[Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed/tree/451e7d084736b1021fb7ea404f0c620b9d785dad) | 固定 `451e7d084736b1021fb7ea404f0c620b9d785dad`；8×H800 demo 为另一 dense 模型，SM90 fused MoE 是 BF16/FP16，不等于目标 block128 FP8 |
| T019 workspace | 同一 Triton-distributed 的 FlashComm planner；[DeepEP pin](https://github.com/deepseek-ai/DeepEP/tree/a56d6156febcd9976e55adc85b5155bfac9f28f8)与[卡片的 vLLM workspace 代码](tricks/T019-bounded-ep-workspace.md) | 原实现已有 rank 去重、部分容量分桶；先判断是否有实际 dispatch，再查 top-10 workspace 高水位 |
| HybridSpec | [arXiv:2611.04055](https://arxiv.org/abs/2611.04055) | 仅一手摘要已核验；作为接受率/真实成本联合调节灵感，不采用未核验全文数字或假设有公开代码 |
| TreeWY | [论文](https://arxiv.org/abs/2608.20961)、[vLLM RFC #54080](https://github.com/vllm-project/vllm/issues/54080) | 作者分支 `b073ed6cacfa1cf5ae23111854729e662a72a6d1`；树宽增加并不保证吞吐，混合状态需完整适配，本轮暂缓 |
| MegaOps / megakernel | [MegaOps v1](https://arxiv.org/html/2611.16248v1)、[归因反证研究](https://arxiv.org/abs/2611.00495)、[MPK](https://www.usenix.org/conference/osdi26/presentation/cheng) | MegaOps 作者代码未核验；MPK 分支 `e801b892ee9b06ba277fb80ca0630c2126c85b1c` 为其他模型 demo。不能宣传完整目标实现 |
| A/F 分离 | SGLang [#45562](https://github.com/sgl-project/sglang/pull/45562) / [#41646](https://github.com/sgl-project/sglang/pull/41646) | 检索时 open；head 分别 `932ba87c338f31888c4da8e823112ad67b0aa2fc` / `4c9cab4ec220ba1c451bb0e5aaec296ebfe59d86`。前者部分路径要求 SM100，后者有同后端负结果，暂缓 |
| CacheBlend / CacheGen | [LMCache 目标模型 recipe](https://docs.lmcache.ai/recipes/qwen3_8_flash_next.html) | 检索日文档明确二者未支持，MTP 未验证；不当成可直接开启的新方案 |

## 4. 如何使用这些证据

源码中存在某个实现，只证明该版本提供了路径。实施 Agent 仍需记录本机实际 dispatch、dtype 与 fallback；本机已有的功能记 `baseline-covered`。论文和其他模型的微基准只帮助选机制，不填入本仓库用户实测结果。

PR 状态、main 文档及模型卡均可能变化，以上锁定的是本次读取时的证据。继续工作先查实际部署和上游差异，不把“当前 blocked”扩展为永久不支持，也不以“已合并”替代完整功能验证。

