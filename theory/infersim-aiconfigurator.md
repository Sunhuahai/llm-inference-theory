# InferSim 与 AIConfigurator：LLM 推理性能建模及配置优化

> 专题研究｜2026-11-24｜v1.0  
> 研究对象：Alibaba InferSim、NVIDIA AIConfigurator。  
> 本文衔接[LLM 推理优化理论总纲](llm-inference-optimization-theory.md)，不绑定特定模型、硬件或精度，不将文献中的性能结果作为本仓库的实测结论。

## 摘要

InferSim 与 AIConfigurator 代表了两种互补的推理性能建模方式。前者从模型结构推导算子工作量、状态规模与数据交换成本，再结合算子效率估计执行性能；后者将真实硬件上的算子时延作为可复用数据，通过执行与服务模型组合预测，搜索满足延迟约束的高吞吐部署配置。

对本项目而言，两项工作的价值在于将三个问题分开：**物理上能够达到怎样的性能、给定实现实际可能达到怎样的性能，以及在多种可行配置中哪一种更优。** 本文介绍工具的实际设计，并基于统一符号给出可以迁移的理论推导。源码事实、论文报告结果和本文的理论扩展分别标明。

## 1. 两项工作的全局位置

~~~text
       模型架构 / 工作负载 / 并行与执行配置
                         |
               算子图及资源需求
                F, D, C, M, S
                         |
           +-------------+-------------+
           |                           |
       InferSim                    AIConfigurator
   解析成本 + 实测效率          实测算子时延数据库
           |                           |
   瓶颈与成本分解            迭代模型 + 服务模式模型
           |                           |
           +-------------+-------------+
                         |
          TTFT / TPOT / 吞吐 / 容量可行性
                         |
             性能边界与配置空间比较
                         |
             真实系统验证 / 请求级模拟
~~~

上图是本文对两种思想的综合抽象，并非二者已提供统一接口。InferSim 可支持分析和候选筛选；AIConfigurator 本身包含配置搜索。最后一层代表独立的验证环节。

| 维度 | InferSim | AIConfigurator |
| --- | --- | --- |
| 开发者 | 阿里妈妈 AI Infra、Future Living Lab | NVIDIA 主导的开源项目 |
| 建模重心 | 资源需求和效率系数 | 真实算子时延及服务配置 |
| 典型问题 | 当前成本由什么组成？改变结构后会怎样？ | 在约束下，哪种可行部署更合适？ |
| 数据依赖 | FLOPs、硬件规格、算子 benchmark/MFU | 算子性能数据库、框架执行模型、硬件数据 |
| 输出 | 阶段 TTFT、TPOT、TGS、成本明细与容量 | 时延、吞吐、可行配置与 Pareto 边界 |
| 理论优点 | 可解释、易研究变量敏感性 | 捕捉实际实现差异、低成本搜索 |
| 共同局限 | 参数化、近似的执行预测；不能自动证明真实线上尾延迟或任务质量 | 同左 |

TGS 表示 tokens/GPU/s。阶段 token 吞吐、输出 token 吞吐、请求吞吐以及 SLO goodput 必须分别定义，不能直接用同一个数比较。

## 2. Alibaba InferSim

### 2.1 公开实现与证据

InferSim 官方仓库：[alibaba/InferSim](https://github.com/alibaba/InferSim)。项目以 Python 实现，公开说明中的重点是从模型超参数预测推理成本及分析瓶颈，算子效率来自微基准数据。[I1]

本节源码核对使用提交 [d12ec47](https://github.com/alibaba/InferSim/tree/d12ec47f83d5896efd77007ad5d2f2ad044c2876)。官方 README 链接的技术报告 PDF 附件本次未能成功读取，因此不将未核实的内容归于该报告。以下分析以公开源码和 README 为依据，并单独给出数学解释。

### 2.2 模型结构如何转换为成本

设单算子形状为 $z$，浮点计算量为 $F(z)$，指定存储层级上的访问量为 $D(z)$，相应精度的峰值算力为 $P$，带宽为 $B$。最简物理下界为

$$
t_{\rm lb}(z)=\max\left(\frac{F(z)}P,\frac{D(z)}B\right).
$$

对 GEMM，令矩阵形状为 $m\times k$ 与 $k\times n$，按乘加两个 FLOPs 计算，

$$
F_{\rm gemm}=2mkn.
$$

引入形状相关的实际计算效率 $\eta(z)$ 后，

$$
t_{\rm gemm}\approx\frac{2mkn}{P\eta(z)}.
$$

InferSim 的 GEMM 路径使用相应的 MFU 查询和计算量换算时延；Attention 与 MoE 路径则进一步比较计算、KV 或权重读取成本。[I2, I3] 这里的 MFU 需要与采用的 FLOPs 口径、精度峰值及算子边界一致。它不是一个可以对全部操作使用的硬件常数。

特别是，如果效率由算子的完整实测时延反推，得到的低效率可能已经包含访存、启动或其他等待。此时再次加上完整访存时延会重复计量。

### 2.3 一个重要的实现原则：实测优先，避免重复计时

核对 InferSim 的 MHA Decode Attention 实现后，可以将其行为概括为：

$$
\widehat t_{\rm attn}=
\begin{cases}
t_{\rm measured},&\text{有相应实测时延},\\
\max(t_{\rm compute},t_{\rm KV}),&\text{没有实测时延}.
\end{cases}
$$

其中 fallback 的理想计算、访存分量可写为

$$
t_{\rm compute}=\frac{F}{P\eta},\qquad
t_{\rm KV}\approx\frac{D_{\rm KV}}B.
$$

实测 Decode Attention 已含 KV 读取时，不再叠加该成本。[I2]

这条关系有两点理论意义：

1. 同一算子完整的实测时间与其内部资源分量是不同层级的量。前者是观测结果，后者用于解释原因。
2. 在允许计算与搬运完全重叠的理想近似下取最大值；若数据依赖使部分操作无法重叠，还需要增加暴露在关键路径上的开销。

因此，MFU 与带宽利用率应被赋予明确的含义：是独立的计算路径效率，还是综合端到端效率？不明确这一点就可能出现重复计量或漏计。

### 2.4 从 Attention、MoE 到一次迭代

一种典型模型层包括：

~~~text
QKV projection
  -> Attention (Prefill or Decode)
  -> Output projection
  -> FFN / MoE
  -> Next layer

MoE with EP:
  dispatch -> expert computation -> combine
~~~

对于固定层结构、无跨组件重叠的近似，可写为

$$
T_{\rm step}\approx
\sum_{l=1}^{L}\left(
t_{{\rm attn},l}+t_{{\rm proj},l}
+t_{{\rm FFN},l}+t_{{\rm comm},l}
\right)+t_{\rm runtime}.
$$

这不是所有实现的精确等式。若两个子图在不同流中执行，必须重新考虑关键路径与资源竞争；不能将 profiler 中所有 GPU kernel 的 duration 直接相加。

在源码中，普通模型将逐层 Attention、投影、MoE/FFN 和通信成本组合，另有双 batch overlap 的近似路径。混合模型由不同类型的层与状态成本构成；部分 shared expert overlap 选项只适用于指定混合模型路径。[I3–I5]

源码还将某些运行时成本设为常数，例如所核对普通模型的 Prefill 调度附加量为 30 ms，Decode 默认附加量为 5 ms（后者可以覆盖）。这些是实现中的假设，不是通用理论常数。真实的主机调度成本可能随并发、token budget、缓存操作、Graph 模式和框架版本变化。

### 2.5 MoE：Batch 与权重复用并非简单线性关系

密集模型的理想权重流量在同一批请求间共享，但 MoE 只触及部分专家。Batch 增大时，被触及的专家并集也可能增大。

本文给出一个简化占用模型：共有 $E$ 个专家，每个 token 独立均匀选择 $r$ 个不同专家。一批 $b$ 个 token 中，某个专家未被选择的概率为

$$
\left(1-\frac rE\right)^b.
$$

因此不同活跃专家数量的期望为

$$
\mathbb E[U(b)]=E\left[1-\left(1-\frac rE\right)^b\right].
$$

若每个专家在选定存储边界读取 $w_e$ 字节，且每个活跃专家权重近似读取一次，则该部分的单位 token 权重流量为

$$
d_{\rm expert}(b)\approx\frac{w_e\mathbb E[U(b)]}{b}.
$$

小 batch 下 $\mathbb E[U(b)]$ 近似 $br$，单位 token 专家权重流量几乎不被摊薄；随专家并集饱和，权重复用机会才增加。非均匀路由应改用各专家选择概率；具体访存还受分片和局部存储影响。这个解析式是本文的通用推导，不声称是 InferSim 全部路由实现的等式。

InferSim 公开的 MoE 路径具有活跃专家估计与权重加载成本计算。[I3, I6] 将专家分布加入模型，是比仅用总参数量或激活参数量更有解释力的方向。

### 2.6 反向约束：目标吞吐如何决定容量预算

在普通单 token Decode 的简化稳态下，若一个副本占用 $g$ 张 GPU，每步有 $b$ 个请求各提交一个 token，步时为 $\tau$ 秒，则

$$
X_{\rm gpu}=\frac{b}{g\tau}.
$$

于是目标每卡吞吐为 $x$、目标步时为 $\tau$ 时，所需副本级活跃 batch 近似为 $b=xg\tau$。等价地，以每卡折算 batch 表示，有

$$
b_{\rm eff,gpu}=x\tau.
$$

例如在单卡或独立 DP attention 的口径下，目标 2000 tok/s/GPU、目标步时 40 ms，需要约 80 个活跃序列的并发生成能力。

在无共享、等长、标准 KV 结构的简化模型中，

$$
\kappa=2Lh_{\rm kv}ds,\qquad
b\ell_{\rm cap}\kappa\le M_0,
$$

其中 $M_0$ 为相应执行组的可用 KV 容量，$\ell_{\rm cap}$ 为预算所采用的状态长度。单卡情形下，给定上述性能目标，有

$$
\kappa\le\frac{M_0}{x\tau\ell_{\rm cap}}.
$$

这是一个有用的必要可行性关系：性能目标反过来限制状态表示大小、可支持上下文与并发规模。并行分片时必须按每设备真实状态副本和存储分片修正，而非直接将设备总容量相加。

InferSim 在未显式指定 Decode batch 时，用目标 TGS 乘目标 TPOT（换算为秒）估算 batch，并检查状态预算；容量路径按输入加输出长度估算，执行 FLOPs 路径则采用输入加约半个输出长度的平均上下文。[I4, I5] 使用 TP 时应特别核对其 batch 与 TGS 除以 TP 的实现口径，不把这一反向关系机械推广到任意并行拓扑。

混合 Attention 的状态可一般化为

$$
M_{\rm state}\approx b(\kappa_{\rm full}\ell_{\rm cap}+m_{\rm recurrent}),
$$

其中 $m_{\rm recurrent}$ 是每序列固定大小的递归状态近似。具体模型可能还包含其他随长度变化的状态。这个分解可用于研究状态结构与吞吐容量之间的关系。

### 2.7 公开证据与限制

官方 README 的示例包括如下阶段吞吐对照：[I1]

| 配置示例 | 阶段 | 实测 TGS | 模拟 TGS | 相对偏差 |
| --- | --- | ---: | ---: | ---: |
| DeepSeek-V3 / H800 | Prefill | 7839 | 9034 | +15.2% |
| DeepSeek-V3 / H800 | Decode | 2324 | 2675 | +15.1% |
| Qwen3-30B-A3B-BF16 / H20 | Prefill | 16594 | 17350 | +4.6% |
| Qwen3-30B-A3B-BF16 / H20 | Decode | 2749 | 2632 | −4.3% |

相对偏差按（模拟值 − 实测值）/ 实测值计算，四舍五入至一位小数。上述为上游展示结果，不是独立复现；不能用四个点推断整个模型或设备空间的预测误差。

InferSim 最适合解释成本结构、研究参数敏感性和给出候选性能估计。对不具备代表性微基准的形状、未知 kernel 路径、复杂混合调度、跨请求缓存命中与随机到达，仍需额外建模。公开路径中的固定运行时预算及平均上下文近似不能直接输出可靠的 p95/p99 或实际 SLO goodput。

## 3. NVIDIA AIConfigurator

### 3.1 研究目标及来源

AIConfigurator 的目标是快速比较大量推理部署配置，降低全量 GPU 压测搜索的成本。核心研究依据为 NVIDIA 团队的 2026 年 1 月预印本 [AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving](https://arxiv.org/abs/2601.06288)；以下论文算法与数据均引用固定的 [v1 HTML](https://arxiv.org/html/2601.06288v1)。[A1]

开源仓库：[ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator)。核对时的仓库 main 提交为 [6cd6f96](https://github.com/ai-dynamo/aiconfigurator/tree/6cd6f9686ab7de6e8fb28171d6e350ee0aa4ed39)。论文版本与持续演进的 CLI/数据覆盖不必完全一致；实际使用应检查所选版本的支持矩阵。

### 3.2 真实算子性能数据库

定义某个基本算子 $o$ 的性能曲面：

$$
t_o=t_o(z;h,e,v),
$$

其中 $z$ 表示算子输入形状、数据类型、布局或通信消息大小，$h$ 为硬件与拓扑，$e$ 为后端，$v$ 为实现版本。论文将 GEMM、Attention、MoE、通信等操作的实测性能组织成可查询数据库，再组合出迭代时延。[A1, A2]

与单纯采用常数 MFU 的区别，是不同形状、后端与执行路径可以有不同的性能曲面。实际算子可能在形状切换时改变 tiling、并行策略和 kernel，出现明显性能台阶。

本文用以下抽象表达查表与解析边界的关系：

$$
\widehat t_o=
\begin{cases}
\mathcal I_o(z),&\text{有效实测覆盖域内},\\
t_{{\rm analytical},o}(z),&\text{无实测覆盖的估计路径}.
\end{cases}
$$

$\mathcal I_o$ 表示由样本构建的插值或查询模型。跨版本、跨设备、跨精度外推必须注明证据等级。上游 CLI 将数据库估计划分为以下模式：[A3]

| 模式 | 含义 |
| --- | --- |
| SILICON | 采用已有实测性能数据 |
| HYBRID | 优先实测；缺失时使用解析与经验修正 |
| EMPIRICAL | 使用理论成本及经验因子 |
| SOL | 使用 Speed-of-Light 理论估计 |

SOL 是理想化理论成本，不是生产性能承诺；HYBRID / EMPIRICAL 的未知误差需与实测支持区分。实际模式、支持范围与行为以对应版本文档为准。

### 3.3 从迭代性能到三种服务模式

论文将固定批执行、混合 P/D 的连续批执行和分离 P/D 三种模式分别建模。[A1，§4.2] 本文使用统一的数学符号解释它们的区别。

**静态模式（Static）。** 令输入长度为 $I$，输出长度为 $O>1$，活跃 batch 为 $b$。若首 token 在 Prefill 结束时产生，理想化时延关系为

$$
TTFT\approx t_P(b,I),
$$

$$
TPOT\approx\frac1{O-1}\sum_{j=1}^{O-1}t_D(b,I+j).
$$

论文采用长度步进近似减少逐 token 查询的开销。平均上下文是一种进一步简化；在长输出或性能曲面非线性时，逐段积分或查询更稳妥。

**聚合模式（Aggregated）。** Prefill 与 Decode 共享执行资源。混合迭代数和纯 Decode 迭代数分别记为 $n_m,n_d$，各自的代表性时延为 $t_m,t_d$，则近似平均步时为

$$
\overline t=\frac{n_mt_m+n_dt_d}{n_m+n_d}.
$$

该式用于说明为何混合负载不能仅用纯 Prefill 与纯 Decode 基准相加。论文对 token budget、混合阶段、生成阶段及 TTFT 采用专门的稳态近似与经验校正（Algorithm 2）。这些校正项不是排队定理，不能直接代表任意真实请求 trace。

**分离模式（Disaggregated）。** Prefill 与 Decode 使用不同资源池。分离的核心问题是两个阶段的服务能力配比，以及状态跨设备转移的额外成本。论文 Algorithm 3 使用阶段容量匹配及经验折减，在可行设备预算中比较资源组合；后续 §4 给出本文的通用推导。

### 3.4 稳态并发、单用户速度与全局吞吐

对固定输出长度 $O$，若 TTFT 与 TPOT 以秒计，单请求平均完成时间可近似表示为

$$
R=TTFT+(O-1)TPOT.
$$

当系统稳定、具有 $N$ 个在途请求的平均并发，并持续补充已完成请求时，根据 Little 定律，

$$
\lambda_{\rm completed}\approx\frac{N}{R}.
$$

因此 $g$ 张 GPU 的平均输出吞吐为

$$
X_{\rm gpu}\approx\frac{NO}{g[TTFT+(O-1)TPOT]}.
$$

对同一固定请求，首 token 之后的平均生成速度为

$$
x_{\rm user}=1/TPOT.
$$

这是论文 §4.2 所采用的吞吐与交互速度关系的统一单位表达。要注意：平均在途请求数 $N$、最大配置 batch 与实际每次迭代的活跃 batch 并非普遍相等；模型将并发等同某个 batch 参数需要明确假设。异长请求下也不能任意将各自平均 TTFT、TPOT、输出长度的乘积用于预测。

这一关系揭示了两个不同的优化目标：单用户生成更快与单位 GPU 完成更多工作。前者和后者可能同时改善，也可能互相制约。

### 3.5 专家负载倾斜为何需要进入性能数据库

在 MoE 中，若设备 $j$ 承担工作量 $W_j$、具有计算能力 $P_j$，则同步执行阶段至少受到

$$
T\ge\max_j\frac{W_j}{P_j}
$$

的限制。均匀平均工作量不能代表慢设备的关键路径。

AIConfigurator 论文使用受控的幂律专家负载构造，使微基准能够测量不同路由倾斜条件下的 MoE 性能（§4.4.1）。[A1] 对本项目而言，这意味着 MoE 性能曲面不仅应以 token 总数为变量，还应考虑专家分布、每专家形状和设备级不均衡。模型的边际性能可能由少数热点专家决定。

### 3.6 配置搜索是约束优化，而不是单指标调参

设部署配置为

$$
c=(g,p_{\rm TP},p_{\rm PP},p_{\rm EP},n_P,n_D,b,C_{\rm ctx},e,\ldots),
$$

其中 $g$ 是资源预算，$n_P,n_D$ 是 P/D 副本数，$C_{\rm ctx}$ 是 context token budget，$e$ 是后端。并非每个组合都可行：并行维度、拓扑、权重驻留、KV、运行时工作区和后端约束可能排除大量组合。

令 $\mathcal C_{\rm valid}$ 为满足这些约束的候选集合。若目标是最大化每卡输出吞吐，则可形式化为

$$
\begin{aligned}
c^*&=\arg\max_{c\in\mathcal C_{\rm valid}}\widehat X_{\rm gpu}(c),\\
&\text{s.t.}\quad \widehat{TTFT}(c)\le \tau_P,\\
&\phantom{\text{s.t.}\quad}\widehat{TPOT}(c)\le \tau_D.
\end{aligned}
$$

在真实生产问题中，还应加入质量、成本及尾延迟约束。论文的核心是以快速的算子数据与服务模型评价候选，而非对每个配置单独完整启动并压测。[A1, A2]

Pareto 边界是另一种等价视角：若配置 A 的所有目标均不差于 B 且至少一个目标更好，则 A 支配 B。所有非支配候选构成 Pareto 集。以下是纯示意数据，并非论文实验：

| 候选 | 单用户 tok/s | 每卡 tok/s | 是否非支配 |
| --- | ---: | ---: | --- |
| A | 80 | 300 | 是 |
| B | 60 | 550 | 是 |
| C | 40 | 800 | 是 |
| D | 50 | 450 | 否，被 B 支配 |

若单用户速度必须至少 60 tok/s，在这些候选中 B 最合适；若放宽至 40 tok/s，则 C 更合适。此例说明没有脱离服务目标的绝对最优配置。

### 3.7 论文验证：数值预测与配置搜索分别评价

AIConfigurator v1 §5 的公开实验给出下列代表性结果：[A1]

| 验证对象 | 结果 | 必须保留的条件 |
| --- | --- | --- |
| 聚合部署 TPOT 误差 | 总体 MAPE 7.8% | 论文测试的模型、H100 SXM 和后端版本 |
| 聚合部署 TTFT 误差 | 各测试组 MAPE 16.9%–22.1% | 排除了 TTFT 大于 1000 ms 的异常点 |
| 分离部署性能 | 吞吐 MAPE 25.49%，生成速度 MAPE 14.94% | 特定 DeepSeek-V3 跨节点实验 |
| 搜索成本 | 506 个 Qwen3-235B 候选，CPU 约 0.84 s | 已有数据库；与包含启动/加载的压测时间比较 |

论文还报告了一个 Qwen3-32B-FP8 案例，每卡输出吞吐由 321.5 增至 648.3 tok/s，单用户生成速度则由 95.9 降至 78.4 tok/s，二者满足该案例的 SLA。详见论文 Table 2。这里比较的是每 GPU 归一化性能；表中聚合执行单位为 TP2，分离配置为 4 个 TP1 Prefill 加 2 个 TP2 Decode 单位，不能将归一化收益无条件解释为相同设备数下的总吞吐增幅。

这些结果不是普适准确度保证。尤其是 TTFT 误差更大且异常点过滤表明：**稳态执行预测与真实排队预测是两个不同的问题。** 搜索无需为每个候选重新进行 GPU profiling，并不意味着建立性能数据库不需要 GPU 实测。

NVIDIA 的部署文档亦明确区分 AIConfigurator 的配置估算、逐请求模拟和真实服务压测。[A4]

## 4. 本文的理论扩展：P/D 资源比例的解析边界

这部分是基于前述工作的独立推导，用于指导优化方向，并非复述某个工具的精确实现。

### 4.1 阶段容量匹配

设固定单 GPU 或归一化资源条件下，Prefill 的稳定能力为 $a$ req/s/GPU，Decode 为 $d$ req/s/GPU。二者均按照相同的输入输出分布折算为请求完成能力。总 GPU 预算为 $G$，分别分配 $g_P$ 与 $g_D$，且假设两阶段容量随资源线性扩展、状态传输不成为限制、无整数与拓扑限制。

$$
g_P+g_D=G,
$$

$$
R_{\rm sys}\le\min(ag_P,dg_D).
$$

最优连续分配的理论条件是两阶段能力匹配：

$$
ag_P^*=dg_D^*.
$$

联立可得

$$
g_P^*=\frac{d}{a+d}G,\qquad
g_D^*=\frac{a}{a+d}G.
$$

相应理想容量为

$$
R^*_{\rm sys}=\frac{ad}{a+d}G.
$$

即每完成一个请求需要的归一化资源时间为 $1/a+1/d$。将两阶段服务能力按照同一工作量计量，是进行正确资源匹配的前提。

### 4.2 一个简单算例

假设 Prefill 每卡能力为 12 req/s，Decode 每卡为 4 req/s，总预算 16 张 GPU。这些为人为设置的教学参数。

理想配比为 4 张 Prefill、12 张 Decode，容量上界为 48 req/s。若平均分配 8/8，则 Prefill 可达 96 req/s，但 Decode 仅为 32 req/s，整体至多 32 req/s；增加 Prefill 资源对这个受限配置没有直接容量收益。

如果跨阶段状态传输另有 36 req/s 的硬上界，则整个系统即使采用理想计算配比，也不能超过 36 req/s。这时继续扩大计算资源无法突破传输限制。

### 4.3 优化方向的边际价值

假设可重新优化连续资源配比，对容量关系取对数导数：

$$
\frac{\partial\log R^*}{\partial\log a}=\frac{d}{a+d},
\qquad
\frac{\partial\log R^*}{\partial\log d}=\frac{a}{a+d}.
$$

如果 Decode 比 Prefill 慢很多，即 $d\ll a$，则提高 Decode 服务效率具有更高的理想边际容量价值。

但如果部署配置被固定、传输受限，或改变某阶段成本会改变另一阶段的负载与资源需求，该导数不再直接成立。真实的 P/D 系统还受整数设备分配、GPU 拓扑、不同并行策略、KV 迁移及调度影响，因此 AIConfigurator 一类工具的候选搜索仍然必要。

## 5. 如何融入统一推理优化理论

### 5.1 三类模型具有不同的证据意义

| 层次 | 核心关系 | 可以支持的结论 |
| --- | --- | --- |
| 物理边界 | FLOPs、数据移动、容量、依赖 | 当前设计可达到的理想边界，优化的必要条件 |
| 实际执行模型 | 算子时延曲面、执行依赖、框架特征 | 给定实现与配置下的校准预测 |
| 服务与决策模型 | 请求并发、P/D 容量、调度、目标约束 | 候选配置排序与可行性；尾部需更细粒度仿真 |

物理边界不等于校准预测；校准预测不等于实测结果。三层不能互相替代。

### 5.2 统一的成本模型及组合原则

对执行图中每个算子 $o$，先定义资源需求向量：

$$
\mathbf d_o=(F_o,D_{o,1},\ldots,C_{o,1},\ldots,M_o).
$$

对可用资源和明确的执行形状，定义理论成本下界 $L_o$ 与经验时延 $\widehat t_o$：

$$
L_o=\max\left(
\frac{F_o}{P_o},
\max_r\frac{D_{o,r}}{B_r},
\max_e\frac{C_{o,e}}{\beta_e}
\right).
$$

算子时延的组合由依赖关系与资源冲突决定。仅在没有并行重叠时可以直接使用串行和；若存在重叠、共享 SM、共享带宽或流水，需使用相应关键路径模型。

稳定生成周期的目标是

$$
X=\frac{\mathbb E[K]}{\mathbb E[T_{\rm cycle}]},
$$

同时必须满足状态容量和任务语义约束。在线服务还需引入请求队列、可变长度和 SLO。对于不在实测支持范围内的优化，解析模型能够给出方向与边界，但不能声称具有已校准的预测精度。

### 5.3 对六大理论分支的贡献

| 理论分支 | InferSim 的启发 | AIConfigurator 的启发 |
| --- | --- | --- |
| 推理成本 | 模型结构、权重与状态的资源账本 | 算子分解和实现相关的时延曲面 |
| 计算–访存平衡 | 计算/MFU/带宽的成本对比 | 形状与后端导致的真实效率变化 |
| 状态复用 | 容量与目标并发的反向约束 | 内存可行性进入候选配置筛选 |
| 自回归生成 | 基准逐 token 成本 | 将额外候选/验证工作纳入算子级模型是可借鉴的扩展方向；需单独建模接受率 |
| 并行与通信 | 权重分片、EP 与通信成本 | 结合实测通信曲面比较并行组合 |
| 推理服务效率 | 阶段吞吐与延迟预算 | 稳态服务近似、P/D 配比和 Pareto 搜索 |

对 Prefix Cache、投机解码、真实 trace 的复用行为、动态排队与 P99，不能仅依据这两项工作的概述假定已经全面支持。需要沿用总纲的专门理论，以及对应版本的数据和实现检查。

### 5.4 最值得研究的四个问题

1. **瓶颈相图。** 在上下文长度、batch、并行度变化时，什么资源下界主导？性能曲面的实现变化是否改变理论分区？
2. **边际优化价值。** 固定任务质量和资源预算，改变一种成本对 TTFT、TPOT、吞吐的灵敏度是多少？优化后瓶颈迁移到哪里？
3. **算子模型的可组合性。** 如何区分已经包含访存/启动开销的实测时间与独立的资源成本？重叠与共享如何改变关键路径？
4. **性能可行域与 Pareto 边界。** 一项优化是否扩大可行域、使更多配置满足 SLO，还是只改善某个固定 operating point？

这四个问题将原有[全景思维导图](diagrams/llm-inference-theory-mindmap.md)中的六个理论分支连接起来。

本文不建议立即重新实现两个开源项目。未来的评估模型可以吸收 InferSim 的资源成本表达，以及 AIConfigurator 的算子数据库、服务配置搜索与性能边界思想。仓库当前仍保持固定精度、固定任务语义，不引入未经验证的量化、压缩或质量退化作为优化收益。

## 6. 阅读源码与参考资料

### InferSim

- **[I1]** [官方 README](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d2f2ad044c2876/README.md)：目标、输入选项、公开模拟结果。
- **[I2]** [layers/attn.py](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d2f2ad044c2876/layers/attn.py) 与 [mfu/mfu.py](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d2f2ad044c2876/mfu/mfu.py)：GEMM、Attention 与效率查询。
- **[I3]** [layers/moe.py](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d2f2ad044c2876/layers/moe.py)：专家计算、权重读取与共享专家 overlap。
- **[I4]** [models/model.py](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d2f2ad044c2876/models/model.py)：权重与 KV 容量、代表性上下文、Prefill/Decode 组合与调度假设。
- **[I5]** [models/hybrid_model.py](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d2f2ad044c2876/models/hybrid_model.py)：混合 Attention 的容量与执行分解。
- **[I6]** [params/params.py](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d2f2ad044c2876/params/params.py)：专家权重与活跃专家相关成本。
- **[I7]** [main.py](https://github.com/alibaba/InferSim/blob/d12ec47f83d5896efd77007ad5d2f2ad044c2876/main.py)：入口、支持的参数和执行路径。

### AIConfigurator

- **[A1]** Xu, T. et al., *AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving*. arXiv:2601.06288v1, 2026. [全文](https://arxiv.org/html/2601.06288v1)。重点：§4.2 服务模式，§4.3 迭代分解，§4.4 数据库，§5 评价。
- **[A2]** NVIDIA, *Removing the Guesswork from Disaggregated Serving*, 2026-03-09. [技术介绍](https://developer.nvidia.com/blog/removing-the-guesswork-from-disaggregated-serving/)。
- **[A3]** [AIConfigurator README](https://github.com/ai-dynamo/aiconfigurator/blob/main/README.md) 与 [CLI User Guide](https://github.com/ai-dynamo/aiconfigurator/blob/main/docs/cli_user_guide.md)：数据库模式和当前支持检查。持续更新页面，具体能力需按版本确认。
- **[A4]** NVIDIA Dynamo, [Size a Local Deployment with AIConfigurator](https://docs.dynamo.nvidia.com/dynamo/cli/disaggregated-serving/sizing-with-ai-configurator)：配置模型与请求级仿真、实际测量的边界。

### 本仓库相关文档

- [统一推理优化理论](llm-inference-optimization-theory.md)
- [全景思维导图](diagrams/llm-inference-theory-mindmap.md)
- [优化收益预测](optimization-benefit-prediction.md)
- [缓存与上下文专项推导](cache-context-ceilings.md)
- [性能预测器](predictor/README.md)

---

**结论：InferSim 将推理成本变成可解释的变量；AIConfigurator 将这些成本与真实执行数据组织成可搜索的性能空间。二者共同表明，理论指导优化的关键不是预测一个孤立的加速比，而是理解性能边界、活跃约束和不同优化之间的耦合关系。**
