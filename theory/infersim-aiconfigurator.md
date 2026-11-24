# InferSim 与 AIConfigurator：使用指南与应用场景

> 业界推理性能工具专题｜2026-11-24｜v2.0  
> 重点：两个工具能解决什么问题、如何安装与使用、如何理解输出，以及在哪些场景值得使用。  
> InferSim 源码基准为 d12ec47；AIConfigurator 命令以 v0.12.0 官方 CLI 指南为基准，并说明迁移至 AISimulate 的安装方式。命令已与公开文档和相关源码核对，但尚未在本仓库的实际 GPU 环境执行；所有上游结果均非本项目实测。

## 目录

- [1. 两个工具如何选择](#1-两个工具如何选择)
- [2. Alibaba InferSim](#2-alibaba-infersim)
- [3. NVIDIA AIConfigurator](#3-nvidia-aiconfigurator)
- [4. 典型应用场景](#4-典型应用场景)
- [参考资料](#参考资料)

## 1. 两个工具如何选择

InferSim 与 AIConfigurator 都支持在不实际启动目标模型服务的情况下估计推理性能。通常只需模型结构、硬件规格、性能数据和工作负载参数，不需要为了每次估计都下载全部权重或占用目标 GPU。

| 研究问题 | InferSim | AIConfigurator |
| --- | --- | --- |
| FLOPs、权重、KV 容量是多少？ | 核心功能 | 支持内存和算子明细 |
| Decode 为什么慢？ | 计算、访存、通信分解 | 实测算子时延与数据来源 |
| 输入长度、batch 改变后性能如何变化？ | 单点估计，可脚本扫描 | 单点估计及配置搜索 |
| 固定 GPU 预算如何选择并行策略？ | 手动设置并比较 | 自动搜索候选 |
| 是否需要 P/D 分离？ | 分别分析两阶段成本 | 原生比较聚合与分离部署 |
| 指定目标 RPS，需要多少 GPU？ | 简化容量估计 | 专门的容量推荐模式 |
| 比较多个推理框架 | 不提供完整框架级比较 | 支持有数据覆盖的后端 |
| 生成部署配置 | 不支持 | 支持 Dynamo、llm-d 等 |
| 预测真实线上 P99 | 不宜直接使用 | 传统 AIC 稳态估计也不能直接保证；后继 AISimulate 提供更细粒度仿真 |

**InferSim 更适合模型成本和瓶颈分析；AIConfigurator 更适合部署配置选择与容量规划。** 二者都不是实际推理引擎，也不能替代最终的真实负载压测。

### 1.1 输入和指标

| 名称 | 含义 |
| --- | --- |
| Model config | 模型层数、Attention、专家数、精度等结构信息 |
| ISL / OSL | 输入与输出 token 数 |
| Batch | 同时参与执行的请求规模；需要核对每卡、每副本或服务级口径 |
| TP / EP / PP / DP | 张量、专家、流水线及数据并行 |
| GPU / System | 被模拟的硬件型号、设备数量及互联条件 |
| Backend | 目标框架与对应性能数据库版本 |
| SLA | 可接受的 TTFT、TPOT 或端到端延迟 |
| TTFT / TPOT | 首 token 时延、平均输出 token 间隔 |
| TGS | tokens/GPU/s，每卡归一化吞吐 |

Prefill TGS 统计输入 token 处理能力，Decode TGS 统计生成阶段能力，不能直接相加。单用户生成速度也不同于 GPU 聚合吞吐。TPOT 以毫秒计且不含 TTFT 时，单用户速度约为 1000 / TPOT。

---

## 2. Alibaba InferSim

官方仓库：[alibaba/InferSim](https://github.com/alibaba/InferSim)  
固定版本：[d12ec47](https://github.com/alibaba/InferSim/tree/d12ec47f83d5896efd77007ad5d5f2ad044c2876)

### 2.1 项目定位及应用场景

InferSim 是纯 Python 的轻量级性能模拟器。从模型结构推导 FLOPs、权重、KV 和通信成本，再结合硬件算力、带宽及已有微基准数据估计性能。

其主要用途是：

| 应用场景 | 使用方式 |
| --- | --- |
| Decode 瓶颈诊断 | 查看 Attention、KV loading、GEMM、MoE、通信的时间明细 |
| Batch Size 设计 | 扫描 batch，分析 TPOT、TGS 与容量上限 |
| 长上下文优化 | 扫描 ISL/OSL，观察状态及 Attention 成本增长 |
| 模型架构设计 | 修改受支持的结构参数，比较 FLOPs、权重与状态容量 |
| 并行策略评估 | 固定资源预算，对比 TP 与 DP Attention + EP MoE |
| 内核优化收益估计 | 改变 MFU 或补充性能数据，进行敏感性分析 |
| 硬件性能比较 | 对已有数据支持的 GPU 型号进行同口径比较 |

其优势是解释慢在哪里，而不是只给出一个预计吞吐量。

### 2.2 安装与第一次运行

核心模拟器不依赖第三方 Python 包，也不需要目标 GPU。可在普通 CPU 主机（包括 Linux 和 macOS）尝试运行；GPU 微基准采集另需真实硬件及相应依赖。

~~~bash
git clone https://github.com/alibaba/InferSim.git
cd InferSim
git checkout d12ec47f83d5896efd77007ad5d5f2ad044c2876

python3 main.py --help
~~~

在项目根目录运行上游提供的示例：

~~~bash
bash example/qwen3-30B-A3B/decode.sh
bash example/qwen3-30B-A3B/prefill.sh
~~~

这两个示例使用仓库内已有模型 JSON，不需要实际下载模型权重。Decode 示例模拟 4 张 H20、每卡 Decode batch 100；Prefill 示例模拟 1 张 H20、最大 Prefill token budget 为 16384。两者分别估计阶段成本，而非同一端到端部署。

### 2.3 关键参数

以固定版本的 [main.py](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d5f2ad044c2876/main.py) 为准。

| 参数 | 含义 |
| --- | --- |
| --config-path | Hugging Face 模型 config.json 路径 |
| --device-type | H20、H800、H200、GB200、PRO5000 |
| --world-size | GPU 总数 |
| --tp-size | TP 大小；此版本大于 1 时 Attention/MoE 均使用 TP，等于 1 时 Attention 使用 DP、MoE 使用 EP |
| --num-nodes | 节点数，用于通信建模 |
| --target-isl / --target-osl | 输入/输出长度，默认 4096/2048 |
| --decode-bs | 指定 Decode batch |
| --max-prefill-tokens | 每卡最大 Prefill token 数，默认 4096 |
| --target-tgs / --target-tpot | 未指定 batch 时，用目标每卡吞吐与目标 TPOT（ms）反推 batch |
| --prefill-attn-mfu | 手动设置 Prefill Attention MFU，范围 (0, 1] |
| --prefill-moe-mfu | 手动设置 Prefill MoE/FFN MFU |
| --decode-scheduler-overhead-ms | 覆盖 Decode 调度时间预算 |
| --enable-deepep | 启用 DeepEP 通信成本模型 |
| --enable-tbo | 启用 two-batch overlap |
| --sm-ratio | TBO DeepEP normal mode 用于计算的 SM 比例 |
| --enable-shared-expert-overlap | 仅特定 Qwen3.5 MoE HybridModel 的共享/路由专家重叠 |
| --use-fp8-gemm / --use-fp8-kv | 切换相应计算或 KV 表示假设；仅在匹配目标精度时设置 |
| --prefill-only / --decode-only | 只估计对应阶段 |

工具的并行方式存在自身实现假设，不等于任意框架的 TP/EP/DP 配置都能一一映射。对固定精度的优化研究，应保持模型、精度、设备总数和负载口径一致。

### 2.4 示例一：分析 Decode 性能瓶颈

~~~bash
python3 main.py \
  --config-path hf_configs/qwen3-30B-A3B_config.json \
  --device-type H20 \
  --world-size 4 \
  --tp-size 1 \
  --target-isl 4096 \
  --target-osl 2048 \
  --decode-bs 100 \
  --decode-only
~~~

该命令回答：指定模型和设备下，batch 为 100 时 Decode 的主要成本是什么？

下表采用上游 README 展示的相同 Decode 示例结果，不是本文重新执行所得。

| 输出字段 | 示例值 | 如何理解 |
| --- | ---: | --- |
| Per GPU params size | 15.19 GB | 单 GPU 权重预算 |
| KV cache space | 60.81 GB | 模型为状态预留的空间 |
| Target per-token KV cache size | 103.79 KB | 此 batch 和长度下允许的单 token KV 大小 |
| Current per-token KV cache size | 96.00 KB | 当前模型估计的单 token KV 大小 |
| Per-token total FLOPs | 9.46 GFLOPs | 每 token 主要算子计算量 |
| Attn core latency | 361.77 μs | Attention 核心时间分解项 |
| KV loading latency | 298.02 μs | KV 读取时间诊断项 |
| Routed experts/FFN latency | 269.28 μs | FFN 或专家计算时间分解项 |
| TPOT | 38.00 ms | 估计的生成步时 |
| Throughput (TGS) | 2632 | 每卡 Decode token 吞吐 |

根据结果选择研究方向：

- Current per-token KV 大于 Target：优先解决显存可行性，而非相信对应吞吐。
- Attention/KV 占较大成本：研究 Attention 实现、状态流量及上下文。
- 专家计算或权重加载成本较大：研究 GroupedGEMM、专家权重复用及路由。
- 通信成本较大：研究 EP/TP、消息粒度、拓扑和计算通信重叠。

不同输出项可能存在重叠，不能把所有打印的微秒数直接相加。已有完整实测 Attention 时延时，该路径不会再叠加 KV 读取时间；缺失实测时才使用计算与访存模型估计。参考 [layers/attn.py](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d5f2ad044c2876/layers/attn.py)。

普通模型的源码还为 Prefill 加入 30 ms 调度假设，为 Decode 默认加入 5 ms（可覆盖）。它们不是实际请求排队时间。不要把模拟 TTFT 解释为线上 P95/P99。

### 2.5 示例二：扫描 Batch 与上下文长度

固定模型、精度、设备和并行方式，扫描 ISL 与 batch。以下脚本保留日志并提取主要指标，适合绘制性能相图。

~~~bash
mkdir -p results/infersim
printf 'isl\tbatch\ttpot_ms\ttgs_tok_gpu_s\tkv_warning\n' \
  > results/infersim/decode.tsv

for isl in 1024 4096 8192 16384; do
  for bs in 16 32 64 100 128; do
    log="results/infersim/isl_$isl-bs_$bs.log"
    if python3 main.py \
      --config-path hf_configs/qwen3-30B-A3B_config.json \
      --device-type H20 --world-size 4 --tp-size 1 \
      --target-isl "$isl" --target-osl 2048 \
      --decode-bs "$bs" --decode-only > "$log" 2>&1; then
      tpot=$(awk '/^TPOT \(ms\):/ {print $NF}' "$log")
      tgs=$(awk '/^Throughput \(TGS/ {print $NF}' "$log")
      warning=0
      if grep -q 'need smaller kvcache' "$log"; then
        warning=1
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' "$isl" "$bs" "$tpot" "$tgs" "$warning" \
        >> results/infersim/decode.tsv
    else
      printf '%s\t%s\tERROR\tERROR\t1\n' "$isl" "$bs" \
        >> results/infersim/decode.tsv
    fi
  done
done
~~~

建议根据输出绘制：

1. Batch–TPOT 曲线：生成延迟如何随并发变化。
2. Batch–TGS 曲线：吞吐提升是否趋于饱和。
3. ISL–Batch 性能热图：标出容量不可行点与高成本区域。

脚本中的日志提取以固定版本普通模型输出为准；对其他版本需核对格式。空值、错误、容量告警及缺失数据不应作为有效配置参与比较。结果可帮助减少后续需要实际 GPU 压测的配置数量。

### 2.6 示例三：估计内核优化的收益

InferSim 支持覆盖 Prefill Attention 或 MoE/FFN 的 MFU。先运行不覆盖 MFU 的 baseline，再逐步改变同一个算子的 MFU：

~~~bash
mkdir -p results/infersim

python3 main.py \
  --config-path hf_configs/qwen3-30B-A3B_config.json \
  --device-type H20 --world-size 1 \
  --target-isl 4096 --max-prefill-tokens 16384 \
  --prefill-only > results/infersim/prefill_baseline.log

for mfu in 0.3 0.5 0.7; do
  python3 main.py \
    --config-path hf_configs/qwen3-30B-A3B_config.json \
    --device-type H20 --world-size 1 \
    --target-isl 4096 --max-prefill-tokens 16384 \
    --prefill-attn-mfu "$mfu" \
    --prefill-only > "results/infersim/prefill_mfu_$mfu.log"
done
~~~

0.3、0.5、0.7 是人为设定的敏感性分析参数，不代表目标 GPU 能达到相应效率，也不必然高于 baseline。

固定计算量 F、算力 P 时，简化计算成本为 F/(P × MFU)。如果提高 MFU 后阶段整体收益很小，可能说明该算子不是主导限制，或其他成本成为新瓶颈。还可以使用 --prefill-moe-mfu 研究 FFN/MoE。此版本没有通用的 --decode-attn-mfu 参数；Decode 的优化应核对对应性能表及源码路径。

### 2.7 示例四：从目标吞吐反推 KV 容量

若目标每卡吞吐为 2000 tok/s、TPOT 为 40 ms，在单卡或独立 DP Attention 的普通逐 token 模型下，需要约 2000 × 0.04 = 80 个活跃序列。

~~~bash
python3 main.py \
  --config-path hf_configs/qwen3-30B-A3B_config.json \
  --device-type H20 --world-size 4 --tp-size 1 \
  --target-isl 4096 --target-osl 2048 \
  --target-tgs 2000 --target-tpot 40 \
  --decode-only
~~~

未指定 decode-bs 时，代码由目标 TGS 与 TPOT 反推 batch。重点检查 Target decode batchsize、Current/Target per-token KV cache size、实际估计 TPOT 和 TGS。

这是目标配置的可行性分析，并不是 SLA 优化器；最终估计值可能达不到目标。在 TP 模式下必须核对每卡与副本级口径。

### 2.8 使用自己的模型与性能数据

| 代码位置 | 用途 |
| --- | --- |
| hf_configs/ | 已有模型配置 |
| config/model_config.py | 模型结构解析 |
| models/model.py | 普通模型成本、容量与阶段时延 |
| models/hybrid_model.py | 混合 Attention 模型 |
| layers/attn.py | Attention 与投影成本 |
| layers/moe.py | 专家及 FFN 成本 |
| mfu/mfu.py | 微基准性能查询 |
| params/params.py | 权重与活跃专家相关成本 |
| bench_data/ | 已有性能数据 |
| kernel_benchmark/ | 真实 GPU 微基准采集 |

切换模型时应检查 Attention、状态、专家、激活及精度的实现是否得到完整支持。能读取 config.json 不代表完整计算图已经被正确建模；不能将新架构的 model_type 改名为旧架构以绕过不支持的问题。

切换硬件或 Kernel 时，需要匹配的硬件规格与性能数据。[kernel_benchmark/](https://github.com/alibaba/InferSim/tree/d12ec47f83d5896efd77007ad5d5f2ad044c2876/kernel_benchmark) 提供 GEMM、GroupedGEMM、MHA、MLA、混合状态与带宽等测试入口。运行这些采集脚本需要真实 GPU 和各自依赖，应先阅读具体脚本，不应将核心模拟器无依赖的特点推广至数据采集。

### 2.9 准确性与适用边界

上游 README 的阶段级 TGS 对照：

| 示例 | Prefill 实测 / 模拟 | Decode 实测 / 模拟 |
| --- | ---: | ---: |
| DeepSeek-V3 / H800 | 7839 / 9034 | 2324 / 2675 |
| Qwen3-30B-A3B-BF16 / H20 | 16594 / 17350 | 2749 / 2632 |

这些是上游报告的示例，不是独立复现，不能推广为所有模型和设备的预测误差。实际性能还取决于形状覆盖、MFU、路由、通信、状态及框架开销。

**适合：离线成本分析、瓶颈解释、参数敏感性分析、GPU 实验前的候选筛选。** 不适合直接保证真实动态负载下的尾延迟和线上服务容量。

---

## 3. NVIDIA AIConfigurator

论文：[AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving](https://arxiv.org/abs/2601.06288)  
原项目：[ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator)  
后继项目：[ai-dynamo/aisimulate](https://github.com/ai-dynamo/aisimulate)

### 3.1 项目定位及使用模式

AIConfigurator 使用模型、硬件与推理框架相关的算子性能数据库，估计执行时延、吞吐及内存需求，并搜索满足服务约束的部署配置。

例如，有 32 张 GPU，输入 3000 token、输出 512 token，要求 TTFT 不超过 1000 ms、TPOT 不超过 10 ms：如何设置 TP、batch、worker 数？聚合部署还是 P/D 分离更好？这是 AIConfigurator 的代表性问题。

| 需求 | CLI 模式 |
| --- | --- |
| 检查模型、系统与后端的数据覆盖 | support |
| 已知配置，估计单点性能 | estimate |
| 固定 GPU 预算，搜索可行高吞吐配置 | default |
| 已知 RPS 或并发需求，估计最少 GPU 数 | recommend |
| 自定义并行候选、异构部署及搜索空间 | exp |
| 快速生成初始部署配置，不完整搜索 | generate |

### 3.2 安装：AIConfigurator 与 AISimulate

截至 2026-11-24 核查，原 AIConfigurator 后续开发已迁移至 AISimulate。原论文方法和兼容 CLI 仍可使用，但新开发应关注后继项目。

[AISimulate v0.12.1 发布说明](https://github.com/ai-dynamo/aisimulate/releases/tag/v0.12.1) 和 [PyPI v0.12.1](https://pypi.org/project/aisimulate/0.12.1/) 已公开。该版本保留 aiconfigurator 兼容命令。以下命令固定版本以避免安装到未来不兼容的版本。

**新环境：安装后继项目。**

~~~bash
python3.12 -m venv .venv-aisim
source .venv-aisim/bin/activate
python -m pip install --upgrade pip
python -m pip install 'aisimulate==0.12.1'

python -m pip show aisimulate
aiconfigurator cli --help
aisimulate --help
~~~

**复现原版 CLI：单独安装 AIConfigurator。**

~~~bash
python3.12 -m venv .venv-aic
source .venv-aic/bin/activate
python -m pip install --upgrade pip
python -m pip install 'aiconfigurator==0.12.0'

python -m pip show aiconfigurator aiconfigurator-core
aiconfigurator cli --help
~~~

Python 采用 3.11–3.13。原 AIC 依赖 native core；平台支持以实际 wheel 为准，优先选兼容 Linux x86-64。macOS ARM64 可检查对应版本是否有匹配的 wheel，但不要假设所有后继版本具有相同支持范围。两个方案应使用独立环境，不必同时安装。

以下六种模式以 [v0.12.0 CLI User Guide](https://github.com/ai-dynamo/aiconfigurator/blob/v0.12.0/docs/cli_user_guide.md) 为准。使用 AISimulate 兼容命令时，先检查各子命令的 --help。离线预测在 CPU 上运行即可；访问远端模型配置可能需要网络，采集真实性能数据库和部署压测则需要对应 GPU。

### 3.3 第一步：检查数据覆盖（support）

~~~bash
aiconfigurator cli support \
  --model-path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm \
  --backend trtllm
~~~

| 参数 | 含义 |
| --- | --- |
| --model-path | Hugging Face ID 或含 config.json 的本地目录 |
| --system | 工具定义的系统名 |
| --backend | trtllm、vllm、sglang 等 |
| --backend-version | 性能数据库版本或支持的版本别名 |
| --systems-paths | 自定义系统/性能数据搜索路径 |

输出分别检查聚合与分离部署。PASS 表示相应 SILICON 数据支持；HYBRID_PASS 表示存在经验或解析 fallback 的估计路径，不等于实测数据充分覆盖。

H800 是 InferSim 的内置设备选项，但不能据此认为 AIC 也有相同的 H800 性能数据库。本文 AIC 示例采用官方文档的 h200_sxm。使用自己的模型与 GPU 时，需要检查实际系统定义、模型、后端、精度与数据库版本；不能拿 H200/H100 的数据冒充 H800 实测。

### 3.4 示例一：预测已知配置（estimate）

~~~bash
aiconfigurator cli estimate \
  --model-path Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm \
  --estimate-mode agg \
  --tp-size 2 \
  --batch-size 64 \
  --isl 2048 \
  --osl 512 \
  --database-mode SILICON \
  --detail memory,time,source
~~~

该命令估计指定 TP2、batch 64、输入 2048、输出 512 的聚合服务，不搜索最佳配置。

| 估计模式 | 用途 |
| --- | --- |
| agg | 聚合服务、连续批处理 |
| disagg | 为 Prefill 和 Decode 指定独立 worker |
| static | 固定形状的两阶段执行分析 |
| static_ctx | 单独 Prefill |
| static_gen | 单独 Decode |

研究 Kernel、Attention、MoE 或通信时，可采用 static 系列和 --detail time,source；研究整个服务配置时应使用 agg/disagg。静态分析不是完整服务吞吐预测。

| 输出字段 | 含义 |
| --- | --- |
| TTFT / TPOT | 首 token 与平均生成间隔估计 |
| Request Latency | 端到端请求时延估计 |
| tokens/s | 部署输出吞吐 |
| tokens/s/gpu | 单 GPU 归一化吞吐 |
| tokens/s/user | 单用户生成速度 |
| seq/s | 请求级吞吐 |
| Concurrency | 当前配置对应的并发指标 |
| Memory (GPU) | 单 GPU 显存占用 |
| Power | 有功耗数据时的估计，缺失或 0 不代表实际零功耗 |

--detail memory 显示权重、KV、激活、NCCL 等显存分项；time 显示算子时延和阶段占比；source 显示实测、经验、估计或混合数据来源。静态模式可进一步查看理想 SOL 对照。

官方 CLI 指南的相应聚合示例报告 TTFT 约 488 ms、TPOT 约 29.1 ms、每卡吞吐约 1076.7 tok/s。这些是上游示例估计值，不是本文运行结果。

### 3.5 示例二：预测指定 P/D 分离部署

~~~bash
aiconfigurator cli estimate \
  --model-path Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm \
  --estimate-mode disagg \
  --isl 2048 --osl 512 \
  --tp-size 2 \
  --prefill-batch-size 4 \
  --prefill-num-workers 2 \
  --decode-batch-size 64 \
  --decode-num-workers 2 \
  --database-mode SILICON \
  --detail memory,time,source
~~~

该例有两个 Prefill worker 和两个 Decode worker，均沿用 TP2，无其他并行维度时共占 8 张 GPU。还可以使用 --prefill-tp-size、--decode-tp-size 等为两个角色分别配置并行策略。

如果要与聚合部署比较，需要固定总 GPU 数量，或者比较相同口径的单位 GPU 吞吐。不能直接把 8 卡分离服务与一个 TP2 聚合 worker 的总吞吐相比。

此功能适合研究已知 P/D worker 比例、各阶段并行度及 batch，检查阶段能力匹配、时延和资源占用。

### 3.6 示例三：固定 GPU 预算搜索（default）

~~~bash
aiconfigurator cli default \
  --model-path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm \
  --backend trtllm \
  --total-gpus 32 \
  --isl 3000 \
  --osl 512 \
  --prefix 0 \
  --ttft 1000 \
  --tpot 10 \
  --database-mode SILICON \
  --top-n 5 \
  --strict-sla \
  --save-dir results/aic_qwen32b
~~~

这个命令固定总预算、模型与负载，搜索聚合和分离部署候选，并保存结果与支持的部署配置。

v0.12.0 默认在搜索阶段执行 TTFT 过滤，TPOT 约束主要用于最终 top-N 选择，因此默认 Pareto 曲线可能包含不满足 TPOT 的点。--strict-sla 将 TPOT 等约束也用于 Pareto 集过滤。若没有可行结果，可以先取消 strict 或放宽目标，检查哪些配置接近可行边界；不可宣称放宽条件后的配置已满足原 SLA。

**如何理解 Pareto 边界**

横轴通常是单用户生成速度，纵轴是单位 GPU 吞吐。非支配方案反映服务体验与总体效率的取舍。

下表仅用于说明如何选配置，不是工具输出。

| 方案 | 单用户 tok/s | 每 GPU tok/s |
| --- | ---: | ---: |
| A | 100 | 300 |
| B | 75 | 500 |
| C | 50 | 750 |

若目标 TPOT 不超过 14 ms（单用户速度至少约 71 tok/s），A、B 满足该条件，B 的每卡吞吐更高；若放宽至 20 ms，C 才进入候选范围。真实选择还应同时检查 TTFT、容量、后端数据覆盖及尾延迟。

工具输出的 best experiment、overall best configuration、各模式 top-N 用于判断在当前估计条件下，哪个部署方式及并行配置更有潜力。

### 3.7 示例四：从业务负载反推 GPU 数量（recommend）

~~~bash
aiconfigurator cli recommend \
  --model-path Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm \
  --target-request-rate 50 \
  --isl 4000 \
  --osl 1000 \
  --ttft 2000 \
  --tpot 30
~~~

此处以 50 req/s 作为负载目标，而非固定 GPU 数量。也可用 --target-concurrency 200 替换 --target-request-rate 50，两者互斥。

输出关注 total_gpus_needed、replicas_needed 与相应并行和服务配置。这适用于 GPU 采购估算、容量规划和预算比较。

推荐的最少 GPU 数是模型假设下的估计，不是生产 SLA 保证。实际部署还需要考虑峰值、长尾输入、缓存状态、故障冗余及相应压测。

### 3.8 示例五：研究 Prefix Cache 与投机生成

固定 ISL/OSL、模型、batch、系统及后端，改变已经缓存的前缀长度：

~~~bash
for prefix in 0 1024 2048; do
  aiconfigurator cli estimate \
    --model-path Qwen/Qwen3-32B \
    --system h200_sxm --backend trtllm \
    --estimate-mode agg \
    --tp-size 2 --batch-size 64 \
    --isl 4096 --osl 512 \
    --prefix "$prefix" \
    --database-mode SILICON \
    --detail memory,time,source
done
~~~

--prefix 是每个请求已缓存的前缀 token 数，不是工具自动预测的实际命中率。该实验适合比较已知复用情景，不能替代真实缓存容量、淘汰、路由、恢复及访问分布的分析。

对支持 MTP 的模型和后端，可以使用 --nextn 和 --nextn-accepted：

~~~bash
# 替换为受支持的 MTP 模型与经过 coverage 检查的系统
MODEL=/path/to/supported-mtp-model

aiconfigurator cli estimate \
  --model-path "$MODEL" \
  --system h200_sxm --backend trtllm \
  --estimate-mode agg \
  --isl 4096 --osl 512 \
  --tp-size 2 --batch-size 64 \
  --nextn 2 \
  --nextn-accepted 1.2 \
  --detail memory,time,source
~~~

模型路径是占位符，1.2 是情景示例而非测量结果。实际使用应输入目标模型与负载下测得的平均接受 draft token 数，范围为 0 到 nextn；不是接受率百分比。模型和数据不支持时，应明确停止或将其作为未校准情景，而不能声称已获得可靠收益。

固定配置对照可用 estimate；如果优化会改变最优 batch、并行度或部署模式，应在同一新条件下重新运行 default 比较。

### 3.9 示例六：自定义搜索（exp）

如果 default 的搜索空间过宽，或者需要限定 TP/EP/PP 候选、指定 P/D 异构系统，可通过 YAML 定义实验。

v0.12.0 使用平坦 Task schema。以下示例按官方 v2 字段组织；需先检查模型和形状覆盖，候选中的每个组合不一定都满足容量、并行整除与拓扑约束。

~~~yaml
exps:
  - disagg_search

disagg_search:
  serving_mode: disagg
  total_gpus: 32
  isl: 4000
  osl: 1000
  prefix: 0
  ttft: 1000.0
  tpot: 40.0
  prefill_model_path: deepseek-ai/DeepSeek-V3
  prefill_system_name: h200_sxm
  prefill_backend_name: trtllm
  prefill_num_gpu_candidates: [4, 8]
  prefill_tp_candidates: [1, 2, 4, 8]
  prefill_pp_candidates: [1]
  prefill_dp_candidates: [1]
  prefill_moe_tp_candidates: [1]
  prefill_moe_ep_candidates: [1, 2, 4, 8]
  decode_model_path: deepseek-ai/DeepSeek-V3
  decode_system_name: h200_sxm
  decode_backend_name: trtllm
  decode_num_gpu_candidates: [4, 8]
  decode_tp_candidates: [1, 2, 4, 8]
  decode_pp_candidates: [1]
  decode_dp_candidates: [1, 2, 4, 8]
  decode_moe_tp_candidates: [1]
  decode_moe_ep_candidates: [1, 2, 4, 8]
~~~

保存为 exp.yaml：

~~~bash
aiconfigurator cli exp --yaml-path exp.yaml \
  --save-dir results/aic_exp
~~~

两个角色使用同一模型，可以指定不同系统。精度默认由模型配置推断，应核对是否与目标部署一致。完整字段参见 [官方 example.yaml](https://github.com/ai-dynamo/aiconfigurator/blob/v0.12.0/src/aiconfigurator/cli/example.yaml) 及同版本高级配置文档。

### 3.10 输出文件与部署配置

通过 --save-dir 保存结果时，常见目录如下，实际文件取决于生成器、版本及部署目标：

~~~text
results/
└── <experiment>/
    ├── agg/
    │   ├── best_config_topn.csv
    │   ├── pareto.csv
    │   └── top1/
    │       └── ...
    ├── disagg/
    │   ├── best_config_topn.csv
    │   ├── pareto.csv
    │   └── top1/
    │       └── ...
    └── pareto_frontier.png
~~~

best_config_topn.csv 保存最佳候选，pareto.csv 保存性能取舍数据。部署文件可能包含框架配置、Dynamo 或 llm-d 配置、Kubernetes manifest 和 benchmark helper。

generate 可以快速产生一个基础配置：

~~~bash
aiconfigurator cli generate \
  --model-path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm \
  --total-gpus 8 \
  --save-dir results/aic_initial
~~~

generate 仅执行基础容量启发式，不进行完整搜索，也不等于全面内存验证。适合作为配置起点，而非已经通过验证的生产部署。

实际运行生成的部署或 benchmark 脚本前，应检查模型路径、镜像、框架版本、设备、网络、缓存与 SLA 口径，避免在 CPU 预测环境直接运行 GPU 部署。

### 3.11 数据库模式及误差

| 模式 | 使用建议 |
| --- | --- |
| SILICON | 优先使用具有真实采集数据的配置；核对硬件、精度、后端、形状和版本 |
| HYBRID | 实测与解析/经验 fallback 混合；适用于覆盖不完整的研究探索 |
| EMPIRICAL | 粗略估计，不视为已校准结果 |
| SOL | 理想资源成本估计，不视为实际执行性能 |

应关注 --detail source 与 warning，了解每项成本的来源。SILICON 不可用时，HYBRID 并不会自动提供同等可信度。

原论文 v1 §5 报告聚合部署 TPOT 总体 MAPE 7.8%；TTFT 各测试组 MAPE 16.9%–22.1%，其中排除了 TTFT 大于 1000 ms 的异常点。该结果有特定实验条件，不能视为任意配置下的误差保证。对于真实排队和尾延迟，需要独立验证。

后继 AISimulate 提供 Replay、Sweeper 以及新的 predict/recommend YAML CLI。真实请求 trace、动态流量或路由策略的模拟应关注其对应版本文档，不能将原 AIC 的稳态模式解释成完整的请求级仿真。

---

## 4. 典型应用场景

本节关注如何选择两个现有工具，不要求重新实现或整合它们。

### 4.1 推理瓶颈分析与内核优化

优先使用 InferSim。固定模型和精度，扫描 ISL 与 batch，查看 Attention、KV、MoE 和通信成本。研究 MFU 改善的条件收益，确定哪些算子和负载区域值得投入 GPU 实验。

AIConfigurator 的 static/estimate 也适合在有真实数据覆盖时核对算子级成本。

### 4.2 并行配置与推理框架选型

优先使用 AIConfigurator default/exp。固定总 GPU 数、模型、精度与 SLA，对有数据覆盖的 vLLM、SGLang、TensorRT-LLM 分别搜索候选，或使用版本支持的 backend auto 模式。比较 Pareto 边界及内存可行性，不只比较不同延迟目标下的最大吞吐。

### 4.3 P/D 分离与资源比例

使用 AIConfigurator default 对比固定预算下的 agg 与 disagg，或用 estimate 对指定 P/D worker 配比进行单点比较。InferSim 可辅助解释两阶段成本为何不同，但独立阶段 TGS 不能直接当作分离服务的端到端吞吐。

### 4.4 GPU 数量与成本规划

使用 AIConfigurator recommend，以目标 RPS 或并发与 SLA 作为输入，获得条件性的最少 GPU 估计。对不同硬件系统重复比较，再考虑设备成本、负载波动及冗余。缺乏目标 GPU 实测数据时，不应直接以峰值算力比例承诺生产容量。

### 4.5 新优化是否值得开发

如果改变的是 FLOPs、状态、访存、MFU 或通信，可优先用 InferSim 判断可能收益与敏感区域。若改变的是支持的前缀复用、投机生成、并行配置或部署模式，可用 AIConfigurator 比较固定配置或重新搜索后的性能。

新架构、Kernel 或调度机制若不在工具支持范围，需修改模型或增加真实性能数据，不能任意修改参数就声称得到了可靠预测。

| 最终需要的结果 | 优先选择 |
| --- | --- |
| FLOPs、权重、KV、通信成本 | InferSim |
| Batch、上下文及 MFU 敏感性 | InferSim |
| 指定部署的时延、显存和吞吐估计 | AIConfigurator estimate |
| 固定预算的部署候选及 Pareto 曲线 | AIConfigurator default / exp |
| 指定业务负载的资源规模估计 | AIConfigurator recommend |
| 可检查的初始部署文件 | AIConfigurator generate / default |
| 动态负载及真实 trace | 对应版本 AISimulate 仿真和实际压测 |

**InferSim 用于理解模型和算子的性能成本，AIConfigurator 用于选择部署配置。二者共同的价值是减少真实 GPU 实验中的盲目参数搜索。**

## 参考资料

### InferSim

- [固定版本官方 README](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d5f2ad044c2876/README.md)
- [CLI 源码](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d5f2ad044c2876/main.py)
- [模型结构解析](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d5f2ad044c2876/config/model_config.py)
- [普通模型成本计算](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d5f2ad044c2876/models/model.py)
- [Attention 成本模型](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d5f2ad044c2876/layers/attn.py)
- [GPU 微基准脚本](https://github.com/alibaba/InferSim/tree/d12ec47f83d5896efd77007ad5d5f2ad044c2876/kernel_benchmark)

### AIConfigurator / AISimulate

- [原论文 v1](https://arxiv.org/html/2601.06288v1)
- [v0.12.0 CLI 使用指南](https://github.com/ai-dynamo/aiconfigurator/blob/v0.12.0/docs/cli_user_guide.md)
- [v0.12.0 官方 README](https://github.com/ai-dynamo/aiconfigurator/blob/v0.12.0/README.md)
- [高级配置](https://github.com/ai-dynamo/aiconfigurator/blob/v0.12.0/docs/advanced_tuning.md)
- [原 AIC PyPI v0.12.0](https://pypi.org/project/aiconfigurator/0.12.0/)
- [迁移指南](https://github.com/ai-dynamo/aiconfigurator/blob/6cd6f9686ab7de6e8fb28171d6e350ee0aa4ed39/docs/aisimulate_migration.md)
- [AISimulate 仓库](https://github.com/ai-dynamo/aisimulate)
- [AISimulate releases](https://github.com/ai-dynamo/aisimulate/releases)
- [AISimulate PyPI v0.12.1](https://pypi.org/project/aisimulate/0.12.1/)

[返回 LLM 推理优化理论总纲](llm-inference-optimization-theory.md)
