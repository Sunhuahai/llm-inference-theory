# 优化收益的理论预测：从算子瓶颈到 TTFT、TPOT 与吞吐量

> 历史模型实例：本文包含 Qwen/H800 特定假设及旧任务案例；通用理论与实测状态应区分。当前研究入口见 [理论导读](../README.md)。

> 推荐先看：[三页优化技术综述](../../visualizations/diagrams/README.md) → [可运行的物理约束预测器](../../models/predictor/README.md)；若重点是 KV/Prefix 命中率、上下文压缩与容量天花板，直接读 [专项推导与 trace 回放](../foundations/cache-context-ceilings.md)。预测器当前用合成示例自测，尚无 H800 实测参数。


> 日期：2026-11-17。状态：`researched`；证据类型：文献、推导与明确标注的假设算例。
> 本文依据仓库基线提交 `1f71162afafb4119762a150b895ccbd1d7447d12` 的约束编写。当前没有用户 H800 实测基线；实际 checkpoint revision、GPU 数量、框架与负载仍见 [环境清单](../../archive/qwen-h800-tricks/environment.yaml) 中的 `unknown`。
> 所有演算数字均非 H800 实测、非官方硬件规格，也不是本项目的收益承诺。固定现有权重、激活、KV/indexer/SSM/PLE 精度，不研究量化、剪枝、蒸馏、缩短回答或关闭所需功能。

## 0. 核心结论与阅读路线

**理论预测不是从“这个 trick 通常快多少”开始，而是从“它能消除当前执行路径上的哪一段成本”开始。**

推荐使用四层模型：

```text
模型与负载 → FLOPs / 内存流量 / 消息 / 状态需求
                ↓
算子性能模型 → 每个 kernel、通信与主机阶段耗时
                ↓
依赖与重叠模型 → prefill / decode 的关键路径时间
                ↓
调度与排队模型 → TTFT、TPOT、输出吞吐与 SLO goodput
```

先用 Amdahl 定律筛掉收益上限太低的方向；再用 Roofline 和通信模型判断局部加速是否物理上可行；最后用固定请求轨迹预测服务指标。Amdahl 与 Roofline 是互补工具，不能互相替代。[^amdahl][^roofline]

本文区分三种结论，不能混写：

| 结论层级 | 输入 | 可以回答 | 不能声称 |
|---|---|---|---|
| 理论界限 | 架构、算法、资源上限与明确假设 | 最乐观能快多少、什么条件下不值得做 | 这个上限一定能达到 |
| 校准预测 | 基线 profile、相似形状微基准、固定 workload | 给定环境的收益情景、瓶颈迁移 | 已完成候选方案的端到端验证 |
| 实测结论 | 同环境交错 A/B、重复统计、质量检查 | 实际收益、回归与适用范围 | 能直接推广到其他并发/长度/拓扑 |

阅读顺序：先读第 1–3 节理解总框架；第 4–7 节用于本模型与各类 trick；第 8–9 节处理吞吐和尾延迟；第 10–12 节直接用于实验与反馈。

## 1. 先明确预测对象，避免指标错位

### 1.1 延迟与吞吐不是一个量

沿用 [Benchmark 协议](../../docs/benchmark-protocol.md)：对请求 i，发送、首个有效生成内容、最后生成内容的时间分别为 s_i、f_i、e_i，实际输出 token 数为 O_i。

$$
\mathrm{TTFT}_i=f_i-s_i,\qquad
\mathrm{TPOT}_i=\frac{e_i-f_i}{O_i-1}\quad(O_i>1).
$$

由定义，对每个满足条件的请求严格有：

$$
\mathrm{E2E}_i=\mathrm{TTFT}_i+(O_i-1)\mathrm{TPOT}_i.
$$

但不能把上述公式中的每项替换成各自 p95：**分位数通常不能相加，平均值也不能随意拆开相关变量的乘积。** 应先生成或重放每请求时间线，再统计分位数。

将 TTFT 作成本分解时，可以写成：

$$
T_{\rm TTFT}=T_{\rm ingress}+W_{\rm queue}
+T_{\rm prefill\to first\ token}+T_{\rm egress}.
$$

这里每项必须对应不重复计时的事件区间；若首 token 已在 prefill 末尾采样，不要再加一次完整 decode。多模态预处理、reasoning 可见性与网络范围沿用原测量协议，不能随优化改变口径。

纯 decode 的规则批次中，若 B 个活跃请求每步各产生一个最终 token，且没有跳步或额外服务停顿：

$$
\mathrm{TPOT}\approx t_{\rm step},\qquad
Q_{\rm out}\approx\frac{B}{t_{\rm step}}.
$$

前式是每个请求的等待时间，**不是** t_step/B；后式才是整个批次的输出速率。动态批处理或 MTP 应改用：

$$
Q_{\rm out}=\frac{\sum_j b_j}{\sum_j t_j},
$$

其中 b_j 只计最终提交的输出 token，不计被拒绝的草稿；完整服务测量还要包含测量窗口内的 prefill、空闲和排空时间。不能用各批次 b_j/t_j 的简单平均代替这个比值。

### 1.2 加速倍数与百分比

定义延迟加速倍数 S_T=T_0/T_1，吞吐倍数 S_Q=Q_1/Q_0。

$$
\text{延迟下降率}=1-\frac1{S_T},\qquad
\text{吞吐提升率}=S_Q-1.
$$

只有固定批大小、相同工作量、没有新瓶颈等条件成立时，才有 S_Q≈S_T。延迟下降 20% 对应 1.25 倍加速，而非“吞吐必增 20%”。

## 2. 第一层：用 Amdahl 定律计算收益上限

### 2.1 从时间守恒推导

设基线时间为 T_0，可优化部分占比为 f，其局部加速倍数为 s；新增开销为 ΔT，记 δ=ΔT/T_0。若这两部分不重叠且其余工作不变：

$$
T_1=(1-f)T_0+\frac{fT_0}{s}+\Delta T,
$$

$$
\boxed{S_T=\frac1{(1-f)+f/s+\delta}}.
$$

因此有净收益的条件是：

$$
\boxed{f(1-1/s)>\delta}.
$$

这是经典 Amdahl 思想在本项目中的成本展开；新增开销由实际方案补入，不应默认零。[^amdahl]

如果某阶段即使完全消除也只占 5%，在其余条件不变、无新增开销时，总加速最多为 1/0.95≈1.053。该上限只约束这里选定的指标与固定执行条件，不是排队系统中所有延迟指标的上限。

### 2.2 一个完整算例

**假设算例，非 H800 数据：** T_0=10 ms，目标阶段占 30%，局部快 2 倍，但引入 0.3 ms 新开销。

$$
T_1=7+3/2+0.3=8.8\ \mathrm{ms},\qquad S_T=1.1364.
$$

总延迟仅下降 12%；固定批次下吞吐可提升约 13.64%，不是 2 倍。该阶段完全消失且零开销的理想上限为 1/0.7≈1.429。

### 2.3 f 应当怎样量

f 不是“某类 kernel 的累计 GPU 时间 / 请求端到端时间”的机械比值。多个 stream、CPU 与 GPU、通信与计算可能重叠；累计 kernel 时间甚至可能大于 wall-clock 时间。应通过时间线识别目标阶段对完成时间的实际约束，并采用第 3 节的依赖模型。Nsight Systems 对 API 开销、launch latency 和 GPU 执行时间有明确区分。[^timeline]

对有重叠的方案，可以先做理想化反事实：在保持资源、依赖规则及正确性的前提下，将目标节点耗时设为零，重算完成时间。该差值是此模型中的可消除时间上限；不能把每个节点单独消除后的收益简单相加。

### 2.4 组合优化为什么不能直接乘

若两个 trick 都减少同一段 launch gap，二者单独各省 0.6 ms，不意味着组合能省 1.2 ms。应该计算同一基线上的组合后成本 T_AB。

$$
S_{AB}=\frac{T_0}{T_{AB}}=S_A\,S_{B\mid A},
$$

但一般不等于 S_A×S_B；第二个因子必须是 **A 已开启后的条件收益**。如果不重叠、没有交互，也应先相加不重复的时间节省，再换算倍数。

## 3. 第二层：Roofline、依赖与资源约束

### 3.1 物理下界不等于时间预测

对一个已经指定算法、形状与数据布局的 kernel j，设 F_j 为 FLOPs、D_j 为必要的 HBM 流量。算力峰值 P_peak 必须匹配实际计算路径，带宽峰值为 β_peak，则理想资源下界为：

$$
\tau_j\geq\max\left(\frac{F_j}{P_{\rm peak}},
\frac{D_j}{\beta_{\rm peak}}\right).
$$

定义算术强度 I_j=F_j/D_j，Roofline 写为：

$$
P_{\rm attainable}\leq\min(P_{\rm peak},\beta_{\rm peak}I_j).
$$

这是排除不可能结果的工具，不是声称 kernel 一定达到这个速度。小形状、低并行度、地址计算、访存依赖和指令发射都可能让实际性能远低于两条 roof。[^roofline][^gpu][^gemm]

必须注意：

- F_j、D_j 是候选执行路径的工作量，不能把基线冗余流量当成所有候选不可消除的流量。
- 不同 dtype 和 Tensor Core / CUDA Core 路径分别建模；不能用 FP8 峰值除所有算子的 FLOPs，也不能误用要求结构化稀疏的宣传峰值。
- 分布式时按 rank 和实际链路计算，不能假设所有卡均匀分担全部工作。
- GPU memory utilization 是容量占用，不是 HBM 带宽利用率；GPU utilization 高也不证明 Tensor Core 已饱和。[^profile]

若确实构造出候选总耗时的严格下界 T_lb,1，则 S≤T_0/T_lb,1。**两个下界的比值 T_lb,0/T_lb,1 本身既不是实测加速，也不自动构成加速上界。**

### 3.2 工程预测需要有效速率

更实用的算子模型是：

$$
\widehat\tau_j=a_j+
\max\left(\frac{F_j}{P_{{\rm eff},j}},
\frac{D_j}{\beta_{{\rm eff},j}}\right).
$$

这里 P_eff、β_eff 与形状、后端、dtype、并行布局有关；a_j 表示该模型尚未解释的设备侧固定成本。CPU launch gap 在时间线上单独处理，不能在 a_j 和 host 时间中重复计算。必要时增加 L2、共享内存或指令吞吐 roof；不能用一个全局“效率 70%”覆盖所有 kernel。[^profile]

有效速率来自相似形状微基准或独立校准，不等于物理上界。用它算出的结果应标为预测，不能标为严格下界。只用某个 kernel 的 F/实测时间反推 P_eff，再用同一个 P_eff 预测自身，会得到循环论证；预测改进需要候选微基准、可迁移的校准或明确的情景假设。

**假设算例：** 同一 kernel 的 F/P_eff=2 ms、D/β_eff=6 ms，暂忽略其他成本。计算加速 2 倍仍得到 max(1,6)=6 ms，几乎无收益；流量减半得到 max(2,3)=3 ms。继续减流量到 1/4 时，时间为 max(2,1.5)=2 ms，瓶颈已经转移到计算，不能继续线性外推。

### 3.3 不要先把所有算子混成一个总 roof

两个串行 kernel，计算/访存时间分别为 (8,1) ms 和 (1,8) ms：

$$
\sum_j\max(t_{{\rm c},j},t_{{\rm m},j})=16\ \mathrm{ms},
$$

而先汇总再取 max 只得到 max(9,9)=9 ms。后者是很松的资源下界，不是该串行程序的耗时预测。融合有机会改变依赖与流量，但不能凭空认为这些阶段已经完全重叠。

### 3.4 关键路径与重叠

将 kernel、host 工作、拷贝、collective 和同步建成有向无环图。对固定调度，除了数据依赖，还要加入设备/stream 的执行次序约束：

$$
\widehat T=\max_{\pi\in\mathcal P}\sum_{j\in\pi}\widehat\tau_j.
$$

这表示最长完成路径；只有节点时长与调度模型成立时才是有效预测。单纯的数据依赖 DAG 若遗漏资源竞争，通常只给乐观结果。并行 kernel 争用同一 HBM 或 SM 时，要重新估计并发有效速率，或显式进行资源受限调度，不能沿用各自独占机器时的时间。

最简单的两阶段重叠模型为：

$$
T=t_C+t_D-\omega\min(t_C,t_D)+t_{\rm sync},\quad 0\leq\omega\leq1.
$$

完全串行时 ω=0，充分重叠时主项为 max(t_C,t_D)。单次请求仍有启动与排空成本；“稳态流水化”不能自动视作“单请求零等待”。对一段已经完全隐藏的通信，仅减少其独占耗时未必降低当前延迟；但减少资源竞争仍可能改变吞吐，需要重新计算。

## 4. 从模型结构计算工作量：不能只看参数量

### 4.1 GEMM 的第一性估算

线性层输入 X∈R^(B×K)，权重 W∈R^(K×N)。按一次乘加计 2 FLOPs：

$$
F\approx2BKN,
$$

$$
D_{\rm ideal}\approx d_wKN+d_aBK+d_oBN+D_{\rm scales}.
$$

d_w、d_a、d_o 分别是既定格式的每元素字节数。此处不改变任何精度；实际 HBM 流量还取决于缓存复用、分块、重读和中间张量。[^gemm]

在权重流量占主导、且所有 B 个 token 复用同一权重矩阵的近似下：

$$
I\approx\frac{2B}{d_w},\qquad
B_{\rm ridge}\approx\frac{d_wP_{\rm eff}}{2\beta_{\rm eff}}.
$$

它解释了为什么增大 batch 可能提高权重复用，但不是完整模型的最优 batch。B_ridge 随 shape/backend 变化；MoE 不同 token 并不都访问同一专家，KV/state 流量也会随负载增加。

### 4.2 当前目标必须按混合架构分账

官方 config 在本次检索时给出 48 层，GDN 与 QSA 按 3:1 排列，512 个 routed experts、每 token 选择 10 个；hidden size 为 2560，routed expert intermediate size 为 640。还有 shared expert、GR、PLE 和 MTP 路径。它不是旧 Qwen3-Next，也不是 48 层同构 dense attention。所有字段仍须与本机 checkpoint 核对。[^config][^qwen]

建议拆成以下账目：

| 组件 | 主要计量对象 | 最易误判的地方 |
|---|---|---|
| 线性投影 / MoE GEMM | 实际形状、激活专家、读权重与中间张量 | 激活参数量不等于真实 HBM 字节数 |
| GDN | recurrent state 读写、更新与 chunk kernel | 长度方向状态固定不等于全部成本与 batch 无关 |
| QSA | indexer、压缩、选择、gather、selected attention、投影 | 稀疏主 attention 不代表 indexer 不随上下文增长 |
| GR | 分支数据、门控、读写、布局转换 | FLOPs 小不代表时间可以忽略 |
| PLE | 本次实际查表、host 内存、PCIe、预取窗口 | 不会每个 token 扫描完整 embedding 表 |
| MTP | 草稿、验证、提交、回滚与状态维护 | 接受长度不是加速倍数 |
| 通信与 host | 每条消息、同步、调度、分词、launch gap | 累计耗时不能直接加到重叠的 GPU 时间上 |

上述结构来自官方配置与架构报告；成本拆分与后续公式是本仓库的建模方案，不是官方宣称的 H800 性能。[^config][^qwen]

### 4.3 MoE：batch 内的专家并集决定权重流量

对于三个投影组成的 gated FFN，单层 routed expert 计算量近似为：

$$
F_{\rm MoE}\approx6Bkhm,
$$

h 是 hidden size，m 是 expert intermediate size，k 是每 token 的 routed experts 数。router、shared expert、激活、dispatch/combine 另计。

设 p_e 是一个 token 选择专家 e 的概率；在不同 token 的路由独立这一假设下：

$$
\mathbb E[U(B)]=\sum_{e=1}^{E}\left[1-(1-p_e)^B\right].
$$

若均匀 top-k，则 p_e=k/E，从而：

$$
\mathbb E[U(B)]=E\left[1-(1-k/E)^B\right].
$$

这是“专家是否至少被访问一次”的指示变量求和，不要求同一 token 的 k 个选择彼此独立。在 E=512、k=10 的**均匀独立路由假设**下，B=1/16/64 时约访问 10/139/367 个不同专家；这不是本模型的实测路由分布。

如果每个被访问专家的权重在该步从 HBM 读取一次，则单层 routed 权重流量可粗估为：

$$
D_{\rm weights}\approx d_w(3hm)\,\mathbb E[U(B)].
$$

此式只是理想复用估计：L2 常驻可减少 HBM 读取，tile 重读可增加；TP/EP 后要按每 rank 的 shard 和专家归属重新计量。实际相关路由可能使均匀独立假设失效，优先使用路由直方图、专家并集与 profiler 流量。

因此不能用“总参数 / 带宽”，也不能用“每 token 激活参数 × B / 带宽”直接给出精确 TPOT。前者可能多算未触达专家，后者可能重复计算批内权重复用。

### 4.4 QSA、GDN 与 PLE 的特殊约束

对普通 dense attention 的单步 decode，一个常见的理想 KV 读取估计为 2BS·n_KV·d·d_KV；**不能把这个式子乘 48 层用于本模型**。QSA 应拆为 indexer 流量、实际 selected KV 读取、索引/gather 与写回；实际选取范围和压缩布局由实现决定。GDN 则独立核算 recurrent state。[^config][^qwen]

PLE 的大表容量与每步传输量是不同问题：每步只计算实际查询、命中、去重和预取的数据。随机 host 查表可能受内存延迟而不只是 DDR 带宽限制；PCIe 拷贝与 GPU 使用之间是否存在足够提前量，也决定预取能隐藏多少成本。

### 4.5 容量约束会改变吞吐收益

每个 rank 都必须满足：

$$
M_{\rm resident}+M_{\rm KV/state}+M_{\rm graph}
+M_{\rm workspace}+M_{\rm reserve}\leq M_{\rm available}.
$$

CUDA Graph、MTP、缓存 checkpoint 或融合 workspace 增大显存后，可能使可用 batch 下降。此时即使单步更快，吞吐也可能更差：

$$
\frac{Q_1}{Q_0}\approx\frac{B_1}{B_0}\frac{t_0}{t_1}.
$$

t_0、t_1 必须分别对应 B_0、B_1 下的实际或预测单步时间；改变 batch 后不能无条件沿用原时间。该式仍只适用于可比较的纯 decode 稳态；真实连续批处理还要重算排队和调度。容量占用、读写流量和同步等待必须分别记录。

## 5. 从理论模型预测具体 trick

### 5.1 CUDA Graph 与 host runtime：只能省掉暴露的间隙

CUDA Graph 的主要机会是减少逐次提交开销，并改善供给 GPU 的连续性。[^graphs]

对确定工作负载，可以用：

$$
\Delta T\approx\Delta G_{\rm exposed}
-\Delta T_{\rm padding}-\Delta T_{\rm fallback}
-\Delta T_{\rm other}.
$$

G_exposed 是基线中真正阻塞后续工作的 CPU/launch gap。不能用“kernel 数量 × 某个通用微秒数”直接承诺端到端收益，因为大量 CPU 提交可能已被 GPU 执行隐藏。Graph shape 覆盖、padding、回退率和显存影响均需计入。一次性 capture/compile 成本与稳态指标分开；若关注冷启动，则另外按该请求集合的使用次数摊销。

T002 与 T011 常共享同一段 host gap；组合时只记一次。

### 5.2 算子融合：减少的不只是 FLOPs

对于串行旧算子与一个新 fused kernel：

$$
\Delta T\approx\sum_j\widehat\tau_{j,0}
-\widehat\tau_{\rm fused,1}+\Delta G_{\rm exposed}.
$$

收益来源可能是中间张量不再写回 HBM、少一次读取、少 launch、改善 tile 复用；FLOPs 可以几乎不变。反面成本包括寄存器压力、spill、occupancy 降低和小形状执行效率变差。**只比较 FLOPs 无法预测 T003/T004。** 先查当前 build 是否已包含融合，再比较真正不同的路径；已覆盖功能的新增收益不能重复计入。

### 5.3 预取与 overlap：收益上限由可重叠窗口决定

若 host 查表/拷贝 D 和 GPU 计算 C 原本串行，充分流水化后的稳态主项从 t_C+t_D 降到 max(t_C,t_D)，理想可省 min(t_C,t_D)。但有三个限制：输入地址必须足够早已知；硬件允许真正并发；重叠不能引入更严重的 HBM/PCIe/CPU 竞争。

**假设算例：** C=6 ms、D=4 ms，串行 10 ms，理想稳态 6 ms，最多约 1.667 倍。若此前已完全重叠，把 D 从 4 降到 2 ms，不会继续把这段的时间从 6 降到 4 ms。冷启动、流水排空和请求间依赖另计。

### 5.4 TP / EP / SP：先算净节省，再考虑卡数

作为解析近似，ring AllReduce 对 p 个 rank、每 rank 逻辑输入 M 字节的耗时可写为：

$$
t_{\rm AR}\approx2(p-1)\alpha
+\frac{2(p-1)}p\frac{M}{\beta_{\rm link}}.
$$

α 表示每轮延迟，β_link 必须匹配该公式的单向有效带宽口径。实际 NCCL 可能使用其他算法或协议，优先以同拓扑、同消息大小测得的曲线替换。NCCL tests 的 `algbw` 与 `busbw` 不是同一个量，不能重复应用换算因子，也不能直接使用整机双向 NVLink 宣传数字。[^nccl]

增大并行度 p→q 的必要净收益条件为：

$$
t_{\rm local}(p)-t_{\rm local}(q)
>t_{\rm comm,exposed}(q)-t_{\rm comm,exposed}(p)
+\Delta t_{\rm other}.
$$

SP 主要减少被重复的局部工作，并引入 gather/scatter 或同步。TP 改变 GEMM 形状与 collective，EP 改变专家分布与 dispatch。不能统一套用“计算除以卡数”。MoE 的等待常由最慢 rank 决定，应保留每 rank 工作量、专家负载和同步时刻，而非只用 rank 平均值。

固定总 GPU 数 G 时，一个副本变快不代表整个服务变快。若每副本用 g 张卡，理想可部署 G/g 个副本，聚合容量还要比较副本数变化；EP 是否复用 TP/DP 进程组以实际框架为准。所有新布局先满足 checkpoint 分片、显存与后端支持约束。

### 5.5 前缀缓存：预测省掉的计算，而不是命中比例本身

对一次请求，设未命中 prefill 时间为 T_miss，命中后的实际处理时间为 T_hit，H 为命中指示，M 为缓存管理开销，则平均节省可写为：

$$
\mathbb E[\Delta T_{\rm prefill}]
=\mathbb E[H(T_{\rm miss}-T_{\rm hit})]-\mathbb E[M].
$$

只有命中请求的长度与成本相近等条件成立，才可简化成 hit_rate×平均节省。不能假定“命中 80% token 就减少 80% TTFT”：后缀仍可能访问已有前缀 KV；还有 lookup、状态复制、排队与网络时间。混合模型还要求与前缀边界一致的 GDN state、QSA/indexer/PLE 状态恢复，不能只算普通 Transformer KV。[^qwen]

T008 主要减少 prefill 工作，不能据此承诺等比例 TPOT 改善；若吞吐改善来自更少的 prefill 干扰，应在混合轨迹中重新预测。缓存的冷/热条件必须与基线一致。

## 6. 原生 MTP：用单位最终 token 成本判定收益

### 6.1 一轮能交付多少 token

设一轮提出 γ 个候选，α_j 是第 j 个候选在前 j−1 个已接受条件下的接受概率。对标准线性 speculative sampling，在每轮还输出一个校正或 bonus token、且忽略 EOS 截尾的条件下：

$$
A_\gamma=\mathbb E[N_{\rm committed}]
=1+\sum_{j=1}^{\gamma}\prod_{r=1}^{j}\alpha_r.
$$

若所有条件接受率均为常数 α，则：

$$
A_\gamma=\sum_{j=0}^{\gamma}\alpha^j
=\frac{1-\alpha^{\gamma+1}}{1-\alpha}.
$$

α=1 时取极限 γ+1。此公式与经典 speculative decoding 分析一致。真实 MTP 实现若没有相同 bonus 规则、存在树状候选、EOS、动态深度或状态限制，应直接按实现统计最终提交 token 数，不强套此式。[^spec]

### 6.2 一轮需要多少时间

令基线单 token 时间为 t_0，候选整轮时间包含 draft、verify、accept/reject、state 管理和调度。它们串行时相加，有重叠时按依赖图计算：

$$
\boxed{S_{\rm decode}\approx\frac{A_\gamma t_0}{t_{\rm round}(\gamma)}}.
$$

因此盈利条件是 t_round<A_γt_0。它是长输出、稳定负载下的单位 token 成本近似，不直接给出每请求 p95 TPOT 或首 token 延迟。混合请求或可变轮次应使用“总提交 token / 总轮次时间”，或在适当稳定性条件下使用 E[N_committed]/E[t_round]，不能简单平均每轮加速比。

**不能假定 verify(γ+1) 的时间等于普通单 token forward。** 验证 shape、MoE 触达专家、KV/state 流量、batch 与额外状态维护都会改变成本。生产引擎中的研究也表明，验证开销与接受长度的分布共同决定 SD 收益，大 batch 的结果不能从 batch=1 外推。[^specbench]

### 6.3 接受率下降与负载变化的算例

**假设算例：** γ=3，条件接受率依次为 0.8、0.65、0.5，则：

$$
A_3=1+0.8+0.8\times0.65+0.8\times0.65\times0.5=2.58.
$$

若 t_0=10 ms、整轮成本 16 ms，则 S≈1.6125，而不是 2.58 倍。另一个假设工作点中，若基线 t_0 仍为 10 ms、整轮成本升至 28 ms，即便这些接受率不变，S≈0.9214，反而变慢；实际切换负载时必须同时重估 t_0 与 t_round。若误把第一个接受率 0.8 用于所有位置，会得到 A_3=2.952，高估可交付长度。

选择更大深度也不是只看 A 增长。对于同一工作点，令新增深度带来的成本与产出增量分别为 Δt>0、ΔA，则扩深度有利当且仅当：

$$
\frac{\Delta A}{\Delta t}>\frac{A}{t}.
$$

这给出了直接的“边际 token / 边际时间”判据。采样语义与质量仍是硬约束；不能用强行接受、改变温度或缩短输出换速度。客户端一个 chunk 可能含多个 token，最终 TTFT/TPOT 按原协议计算，不能把整轮产出误作多个零延迟事件。

## 7. 现有 12 个 trick 应当怎样预测

下表不新增优化承诺，只把已有卡片映射到理论模型；实现支持仍以各卡片与实际 build 为准。

| Trick | 主要变化 | 预测前最少需要什么 | 上限或回退风险 |
|---|---|---|---|
| [T001 分块 prefill](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T001-prefill-scheduling.md) | 混合 batch 形状、调度顺序 | 混合 step 成本曲面、到达轨迹、chunk 策略 | 可能改善 decode 尾部却增加 prefill 总成本 |
| [T002 CUDA Graph](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T002-cuda-graphs.md) | 暴露的提交间隙 | 时间线、shape 命中、padding/fallback | 已被隐藏的 CPU 开销不能再次扣除 |
| [T003 QSA prepare](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T003-qsa-prepare.md) | 中间流量、launch 与布局 | 旧/新路径是否存在、shape、F/D | baseline-covered 时无新增实现收益 |
| [T004 QSA projection](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T004-qsa-projection.md) | 小 GEMM 效率、融合 | 每种 TP/batch 的形状与校准速率 | 寄存器、spill、瓶颈迁移 |
| [T005 GR/PLE SP](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T005-gr-ple-sequence-parallel.md) | 重复工作与新增通信 | 每 rank 工作量、collective 曲线 | 节省的局部工作须超过新增暴露成本 |
| [T006 PLE 预取](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T006-ple-prefetch.md) | host/PCIe 等待与重叠 | 每步实际查表量、预取窗口、NUMA | 已隐藏部分不能重复记收益 |
| [T007 并行拓扑](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T007-parallel-topology.md) | rank 形状、消息、副本数 | 合法进程组、同总卡数、拓扑 | 单副本更快而聚合容量下降 |
| [T008 混合缓存](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T008-hybrid-prefix-cache.md) | prefill 重算与状态恢复 | 命中长度分布、恢复/管理成本 | 命中率不等于 TTFT 改善率 |
| [T009 原生 MTP](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T009-native-mtp.md) | 每轮产出及验证成本 | 分位置接受率、round 时间、内存 | A_γt_0≤t_round 时无净收益 |
| [T010 MoE 通信](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T010-moe-communication.md) | dispatch/combine 与 straggler | 消息分布、每 rank 专家负载、重叠 | 不能只看平均 rank 或孤立 NCCL 时间 |
| [T011 host runtime](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T011-host-runtime.md) | 分词、调度、CPU gap | CPU 时间线及实际阻塞关系 | 与 Graph/fusion 节省重复计数 |
| [T012 P/D 分离](../../archive/qwen-h800-tricks/weeks/2026-W47/tricks/T012-prefill-decode-disaggregation.md) | 阶段容量、排队、状态传输 | 同 GPU 总预算、阶段服务曲线、状态大小 | 传输/分配不平衡抵消去干扰收益 |

## 8. 第三层：调度、排队与吞吐为何改变结论

### 8.1 到达率固定时，容量上升不一定增加实际吞吐

设固定任务分布下的请求到达率为 λ，可持续处理容量为 μ。在稳定、无丢失且 λ<μ 的系统中，长期完成速率就是 λ。把 μ 从 100 提高到 120 requests/s，在输入只有 20 requests/s 时，不会凭空多完成请求；改善体现为延迟、余量或支持更高 offered load 的能力。

纯 decode 的 B/t_step 是一个工作点的输出能力，不等于生产流量下的实测吞吐，更不等于有 TTFT/TPOT SLO 的 goodput。预测吞吐收益前必须写清是“固定负载的已完成吞吐”“饱和容量”还是“满足 SLO 的最大容量”。

### 8.2 小幅服务加速可以放大为大幅排队改善

仅作机理说明，考虑泊松到达、指数服务、单服务器、FCFS 的 M/M/1 队列，稳定条件为 λ<μ：

$$
\mathbb E[T_{\rm response}]=\frac1{\mu-\lambda}.
$$

**假设算例：** λ=90 requests/s，μ 从 100 提高到 120，平均响应时间由 100 ms 降到约 33.3 ms。服务容量提高 20%，平均响应改善却达到 3 倍。这不是对 LLM TTFT 的数值预测，只说明接近饱和时排队具有非线性。[^queue]

更一般的 FCFS M/G/1，在独立同分布服务时间 X、泊松到达、有限二阶矩且 ρ=λE[X]<1 的条件下：

$$
\mathbb E[W_q]=\frac{\lambda\mathbb E[X^2]}{2(1-\rho)}.
$$

它说明服务时间波动也重要；不能只优化平均 kernel 时间而忽略长停顿。但连续批处理、抢占、共享 GPU 和 prefix cache 使真实 LLM 服务不满足这个简单队列的全部假设。上述公式用于解释，不直接用于宣称 p95/p99 或精确 SLO 容量。[^queue]

### 8.3 Chunked prefill 要预测混合 step，不能直接相加

建议建立：

$$
t_{\rm step}=g(B_d,N_p,\mathcal S,\text{routing},\text{cache/state},\text{backend}),
$$

其中 B_d 是 decode 请求数，N_p 是本步 prefill token 数，S 集合记录上下文长度。混合 step 可能改变权重复用、GEMM 形状、访存与同步，所以通常不等于“纯 prefill 时间 + 纯 decode 时间”。

Sarathi-Serve 的研究说明，分块 prefill 可以改变吞吐和 decode 停顿的权衡；其论文收益不能直接迁移到本模型与 H800。[^sarathi] 本项目需要使用同一到达轨迹，重放 admission、token budget、chunk 选择、内存分配与完成事件；每次 step 用经过校准的 g 估时。

### 8.4 P/D 分离的容量上限

设 prefill/decode 分别部署 n_p、n_d 个副本，每副本分别使用 g_p、g_d 张 GPU，固定预算满足：

$$
n_pg_p+n_dg_d=G.
$$

μ_p、μ_d 是相同任务分布下每副本的阶段完成能力，统一为 requests/s；每请求需传输状态 M_state。在理想专用网络假设下，可用以下容量筛选式：

$$
\lambda_{\rm cap}\leq\min\left(n_p\mu_p,n_d\mu_d,
\frac{\beta_{\rm transfer}}{\mathbb E[M_{\rm state}]}\right).
$$

它未包含排队、同步、共享网络争用和 SLO，因此仍是乐观筛选式。传输对象不能只算普通 KV，必须覆盖引擎恢复此混合模型所需的全部状态；两端 dtype、布局及传输支持也须确认。DistServe 提供了去除阶段干扰与联合分配资源的研究依据，但不能据此假定当前框架支持本模型完整状态迁移。[^distserve]

### 8.5 哪些预测需要离散事件模拟

涉及 T001、T008、T012 或高负载 MTP 时，建议在逐算子模型之上做事件重放。输入同一份 arrival trace、长度分布、缓存条件和 GPU 预算，产生每请求首 token、末 token、错误/超时与资源占用，再按协议统计指标。

Vidur 是“算子 profile + 性能预测 + 服务仿真”的代表性一手研究。[^vidur] 本仓库借鉴方法，不宣称它开箱即用支持 Qwen3.8-Flash-Next；GDN/QSA/PLE/MTP 与实际 scheduler/state 管理需要独立建模和验证。没有这些能力时，保留边界分析，不输出伪精确的尾延迟。

## 9. 一次可复核的组合收益演算

以下全部为**互不重叠的假设成本**，仅示范如何计算；不是本机 profile，也不代表优先级。

| 单步成本 | 基线 ms | 候选 ms | 假设依据 |
|---|---:|---:|---|
| GEMM 与其他 GPU 工作 | 5.0 | 5.0 | 不变 |
| QSA prepare 设备工作 | 1.0 | 0.6 | 融合省 0.4；不含 launch gap |
| 暴露的通信等待 | 2.0 | 1.5 | 省 0.5；已排除隐藏部分 |
| 暴露的 PLE 等待 | 1.0 | 1.0 | 不变 |
| 暴露的 host/launch gap | 1.0 | 0.3 | Graph 省 0.7 |
| **总计** | **10.0** | **8.4** | 假定没有额外成本与瓶颈切换 |

得到 S=10/8.4≈1.1905，即单步延迟下降 16%。若固定 B=16、纯 decode 且持续有足够请求，则输出能力从 1600 升到约 1904.8 tokens/s，提升 19.05%。

再考虑一个确定长度的请求，TTFT=100 ms 不变、O=128，则：

$$
T_{\rm E2E,0}=100+127\times10=1370\ \mathrm{ms},
$$

$$
T_{\rm E2E,1}=100+127\times8.4=1166.8\ \mathrm{ms}.
$$

端到端加速只有约 1.1742。另作一个明确的容量情景：若 Graph 额外占用显存导致 batch 从 16 降到 12，且新 batch 下的单步时间仍为 8.4 ms，则纯 decode 吞吐比为 (12/16)×(10/8.4)≈0.8929，反而下降约 10.71%。这说明“kernel 更快”“TPOT 更低”“整体吞吐更高”必须分开验证。

## 10. 怎样从零数据走到可信预测

### 10.1 最小校准清单

没有环境和 profile 时，先给符号公式与情景分析，不填入猜测的 H800 参数。以下由有机器权限的实施 Agent 收集，本次文档编写没有启动 GPU 作业。

| 需要的量 | 获取方式 | 进入哪一层 |
|---|---|---|
| checkpoint、实际 dtype、软件 SHA、H800 SKU/数量/拓扑 | 环境与加载日志 | 适用性、资源与容量约束 |
| 基线每请求时间戳、输出数、到达序列 | 固定 benchmark 与脱敏 trace | TTFT/TPOT、队列与吞吐 |
| kernel/host/copy/collective 时间线 | 小范围 Nsight Systems profile | 重叠与关键路径 |
| 实际 GEMM shape、DRAM 流量与主要限制 | 选定 kernel 的 Nsight Compute | Roofline 与校准速率 |
| 相同 shape/dtype/backend 的候选微基准 | 独立小对照 | 局部 s、固定成本、回退 |
| NCCL 各消息大小/拓扑成本 | 同进程组微基准 | α、β 与通信曲线 |
| 专家命中与每 rank 负载 | 路由/调度统计 | MoE 复用和 straggler |
| cache 命中长度、状态恢复、峰值内存 | 引擎统计与 profile | 前缀缓存、容量、抢占 |
| MTP 分位置接受率与 round 分项成本 | 实际业务与相同采样 | A_γ 和单位最终 token 成本 |

微基准不一定需要大规模 sweep。先选择最相关的两个负载点，再增加一个不同 batch/长度的留出点；覆盖的范围之外不宣称可靠外推。具体点先服从显存与模型支持，不能把建议点当成本机可运行保证。

### 10.2 Profiler 的数字不能直接当服务结果

Nsight Compute 的 replay、cache 控制和采集会影响时序，不能拿开启重 profiling 的 host wall time 当正常 benchmark。[^profile] 用独立的无重 profiler 运行确定正式指标，短 profile 解释机制；Nsight Systems 采样也需评估扰动。校准记录同时保存工具版本、采样范围和预热状态。

### 10.3 留出验证和误差

对未参与拟合的工作点，定义：

$$
\epsilon_T=\frac{\widehat T-T_{\rm measured}}{T_{\rm measured}}.
$$

报告有符号误差与绝对误差，分别检查 prefill、decode、TTFT、TPOT 与容量。只有基线吻合不够：模型还必须能解释至少一个候选的变化趋势，否则它可能只是拟合了基线总时间而没有识别瓶颈。

错误优先排查：漏算状态/中间流量、错误的 dtype 峰值、重复计通信、shape 外推、路由不均衡、缓存冷热差异、Graph fallback、客户端限流和输出分布变化。

### 10.4 用情景区间，而非未经校准的置信区间

未知局部效率、接受率或命中率时，为参数设定有依据的保守/中性/乐观情景，并让 baseline/candidate 共享同一组环境参数计算差值。这些是**情景区间**，不自动具有“95% 概率”含义。

有重复实验后，可按独立运行或保留相关性的请求块做 bootstrap；不要把同一服务批次中的相关请求当完全独立样本。统计波动与模型误差分别报告，不能用更多请求样本消除结构性建模错误。

## 11. 交给实施 Agent 的预测记录模板

每个候选先写一份预测，再实验，避免看完结果回填假设。可放在当周 `results/<trick-id>-prediction.md`，不要把本文假设数字写入实测 CSV。

```yaml
prediction_version: 1
status: hypothesis
evidence_origin: hypothesis
trick_id: Txxx
baseline_id: unknown
env_id: unknown
workload_id: unknown
model_revision: unknown
engine_git_sha: unknown
gpu_budget: unknown
metric_target: unknown  # TTFT / TPOT / output capacity / SLO goodput
load_mode: unknown
cache_condition: unknown

mechanism:
  changed_cost: unknown
  unchanged_cost: unknown
  critical_path_or_resource: unknown
  added_overhead: unknown
  memory_capacity_effect: unknown

model:
  equations_and_sections: unknown
  inputs_with_evidence: unknown
  assumptions: unknown
  counterfactual_upper_bound: unknown
  conservative_scenario: unknown
  central_scenario: unknown
  optimistic_scenario: unknown
  predicted_next_bottleneck: unknown
  invalidation_conditions: unknown

validation:
  one_variable_ab: unknown
  holdout_workload: unknown
  quality_gate: unknown
  ttft_tpot_error_timeout_constraints: unknown
  measured_result_ref: unknown
  prediction_error: unknown
  rollback: unknown
```

预测模板与 [现有反馈模板](../../archive/qwen-h800-tricks/templates/feedback.md)、[结果规范](../../docs/benchmark-protocol.md) 并用。`unknown` 不填零；未经实验不升级为 `validated`。如果只能推导上限，就明确只给上限。

## 12. 本项目现在应该采用的策略

当前缺真实基线，因此尚不能有依据地给出“本机优化后提升 20%–40%”一类范围。可以立即确定的，是每种 trick 需要什么数据、如何计算净收益以及在哪个条件下应停止投入。

第一步，建立一个可信基线并识别暴露的成本。第二步，从已有 T001–T013 中只选一项，计算理想上限、合理情景与新增成本；对有重叠的方案先重算关键路径。第三步，实施单变量 A/B，用偏差更新模型，再判断下一个瓶颈。

理论最有价值的输出不是一个孤立百分比，而是可证伪的判断，例如：“当前 launch gap 至多占 4%，仅消除该间隙不可能在固定执行条件下获得 20% TPOT 改善”；或“该 MTP 深度需要每轮提交超过 t_round/t_0 个 token 才有净收益”。

**用可消除成本确定上限，用微基准确定可达性，用调度模型确定服务收益，用真实 A/B 决定结论。**

## 参考资料与使用边界

以下均为一手论文、作者所属机构页面或官方技术文档，检索日期为 2026-11-17。滚动页面的内容可能变化；实施实验必须另行固定实际 release/git SHA、checkpoint revision 与工具版本。本文的假设算例、MoE 专家并集推导、组合成本表与项目流程为本仓库分析，不是来源对用户 H800 的测量结论。

[^amdahl]: Gene M. Amdahl, [Validity of the single processor approach to achieving large scale computing capabilities](https://doi.org/10.1145/1465482.1465560), AFIPS 1967。用于固定工作量下的部分加速与整体上限。
[^roofline]: Samuel Williams, Andrew Waterman, David Patterson, [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html), UC Berkeley, 2008。用于计算/内存资源约束；不是单靠硬件峰值即可准确预测服务时间的保证。
[^gpu]: NVIDIA, [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)。用于算术强度与低并行度/延迟限制；其中示例硬件参数不作为本项目 H800 参数。
[^gemm]: NVIDIA, [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。用于 GEMM FLOPs、内存流量、形状与 tile 效率。
[^profile]: NVIDIA, [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。用于 roofline、计数器、replay 与测量扰动边界。
[^timeline]: NVIDIA, [Understanding the Visualization of Overhead and Latency in NVIDIA Nsight Systems](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)。用于区分 CPU wrapper、launch latency、拷贝与设备执行。
[^graphs]: NVIDIA, [CUDA Graph Quantitative Benefits](https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/quantitative-benefits.html)。用于 Graph 减少提交开销的机制；本文件不直接采用其通用每 launch 时间估计。
[^config]: Qwen, [Qwen3.8-Flash-Next-FP8 official config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json)。用于架构字段，不代表用户 checkpoint 已核对。
[^qwen]: Qwen, [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/html/2608.30320v1), 2026，固定引用 v1。用于 GDN/QSA/GR/PLE/MTP 结构；不迁移其 kernel 加速比或架构对照为本项目调优收益。
[^nccl]: NVIDIA, [NCCL tests: Performance](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。用于操作时间、algbw/busbw 和 collective 数据量；实际算法与小消息延迟要本机校准。
[^spec]: Yaniv Leviathan, Matan Kalman, Yossi Matias, [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html), ICML 2023，特别是 §3 的产出与 wall-time 分析。本文放宽恒定接受率和免费验证假设，保留正确性前提。
[^specbench]: Xiaoxuan Liu et al., [Speculative Decoding: Performance or Illusion?](https://arxiv.org/html/2601.11580v1)，固定引用 v1。用于接受长度分布、验证成本及 batch 依赖；原实验模型/硬件不是用户当前部署。
[^sarathi]: Amey Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal), OSDI 2024。用于 chunked prefill 与调度权衡。
[^distserve]: Yinmin Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), OSDI 2024。用于 P/D 干扰、资源分配与 goodput。
[^vidur]: Amey Agrawal et al., [Vidur: A Large-Scale Simulation Framework For LLM Inference](https://arxiv.org/abs/2405.05465), MLSys 2024。用于 profile 驱动的预测与离散事件仿真思路，不构成目标混合架构已支持的证明。
[^queue]: Richard C. Larson and Amedeo R. Odoni, [Urban Operations Research, §4.7: The M/G/1 Queueing System](https://web.mit.edu/urban_or_book/www/book/chapter4/4.7.html), MIT 在线教材。用于单服务器排队的稳定性与均值关系；M/M/1 是本文说明用的特殊情形，不用于直接预测 LLM 尾分位数。
