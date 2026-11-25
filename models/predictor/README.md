# H800 推理优化收益预测器（v0.2：关键路径 / 资源容量 + Prefix / 上下文）

> **状态：可运行的物理约束 + 基线校准情景模型；尚无用户 H800 实测数据。** 本目录的数字案例全部是 **SYNTHETIC**，不能引用为加速成绩。

本模型要回答的不是“某个 kernel 快了多少”，而是在**同一个 checkpoint、精度、GPU 预算和请求分布**下：

1. 低负载平均 TTFT 能减少多少？
2. 低负载平均 TPOT 能减少多少？（多 token chunk 需换算成每 token）
3. 固定负载形状下，饱和输出 tokens/s 容量能增加多少？
4. 新瓶颈在哪里？哪些预测缺乏证据？

配套的三页技术综述见 [可编辑 draw.io 图解](../../visualizations/diagrams/README.md)；完整理论推导见 [原理论文档](../../theory/optimization/benefit-prediction.md)；缓存与压缩的专门推导见 [前缀状态复用、上下文压缩与性能天花板](../../theory/foundations/cache-context-ceilings.md)。

**两个相互补充的预测模块：**

| 模块 | 负责什么 | 不负责什么 |
|---|---|---|
| [`predict.py`](predict.py) + [`model.py`](model.py) | 不同内核/调度改动对 TTFT、TPOT 与饱和输出 tok/s 的情景预测 | 不会凭空生成业务缓存命中率或语义压缩质量 |
| [`cache_context.py`](cache_context.py) | 按序 trace 重放 Prefix 命中、LRU 驱逐、压缩前后缓存键，输出 TTFT 上界与质量 gate | 不预测并发 p99、块级混合状态缓存或饱和 TPS |

**组合原则**：以真实冷/热 trace 为起点逐项应用干预。缓存与上下文压缩会重写实际 Prefill 负载、键分布与资源需求，**不能将两个模块的 speedup 直接相乘**。

多个已尝试改动应按[优化耦合](../../theory/optimization/coupling.md)整理基线、A、B、AB 及父版本，再按[模型验证](../../experiments/model-validation/README.md)检查收益方向和候选排序；实际项目的整理与选读方法见[项目指导](../../docs/project-optimization-guide.md)。

## 30 秒运行

从仓库根目录执行，不需要安装第三方 Python 包：

~~~bash
python3 models/predictor/predict.py \
  models/predictor/examples/synthetic-baseline.json --format markdown

python3 models/predictor/predict.py \
  models/predictor/examples/synthetic-baseline.json --format json

python3 models/predictor/cache_context.py \
  models/predictor/examples/synthetic-cache-context.json --format markdown

python3 -m unittest discover -s models/predictor/tests -v
~~~

使用自己的数据时，复制 JSON 到一个新实验文件，填入真实基线与 profile；不要把演示数据改名后冒充真实结果。CLI 可用 `--output result.json` 写入可复查预测。

不运行代码也可以先看 [普通优化合成预测](examples/synthetic-prediction.md)与[缓存/压缩合成预测](examples/synthetic-cache-context-result.md)；它们的输入契约分别见 [普通场景 schema](SCHEMA.md)和[缓存/压缩 trace schema](CACHE_CONTEXT_SCHEMA.md)。

## 四层模型

### A. 关键路径：先校准 TTFT、TPOT

对一批配对的低负载请求，按**均值**拆账：


$$
T_{\rm TTFT,0}=T_{\rm fixed}+T_{\rm prefill,0}+T_{\rm first,0},
\qquad T_{\rm TPOT,0}=T_{\rm decode,0}.
$$


**不能把 p50/p95 直接相加**。每个阶段再拆成互不重复计费的 block；其中串行时间 `s`、计算时间 `c`、通信时间 `m` 与已实现的重叠比例 `η`：


$$
t_b=s_b+c_b+m_b-\eta_b\min(c_b,m_b),\qquad0\le\eta_b\le1.
$$


当 `η=0`，两条 lane 串行；`η=1` 时其合计为 `max(c,m)`。这只是两条 lane 的关键路径近似，不是任意异步 DAG 的完整模拟。

一个候选可改变三个局部加速比、重叠比例及新增开销；先算不含新增开销的结构部分：


$$
t'_{b,\rm structural}={s_b\over k_s}+c'_b+m'_b
 -\eta'_b\min(c'_b,m'_b),
$$



$$
c'_b=\max(c_b/k_c,\ell_c),\qquad
m'_b=\max(m_b/k_m,\ell_m).
$$


其中 `ℓ` 是用户提供的**情景时间下限**。`max(FLOPs/P_eff, bytes/BW_eff)` 使用对应形状的有效速率时，只能提供校准情景估计，不能称为严格资源下界；只有速率确为适用范围内的可证明上界、工作量口径匹配时，相应比值才构成条件性时间下界。现有 `*_floor_ms` 字段不验证这类证明，必须在实验记录中注明来源。峰值也不能当作可持续有效速率。新 [解析参考](../analytic/README.md)对这两种输入语义作显式区分。

对每个阶段以真实低负载均值进行乘法校准：


$$
\widehat T'_{\rm stage}=
T_{\rm stage,0}^{\rm measured}\,
{\sum_b t'_{b,\rm structural}\over\sum_b t_b}
+\sum_b h_b.
$$


零改动严格重现基线。局部 2× 会自然受到 Amdahl、重叠和新增开销限制。

### B. MTP：一轮成本除以真正交付的 token

仅对**一次 target step 恰好产出一个 token 的基线**，如果每一步接受 draft 的条件概率近似为常数 `p`，深度 `d`，则


$$
E[K]=\sum_{i=0}^{d}p^i,\qquad
\widehat{\mathrm{TPOT}}_{\rm MTP}=
{T'_{\rm decode}\,k_{\rm verify}+h_{\rm draft}\over E[K]}.
$$


这是情景近似。真实系统必须实测接受长度分布、verification 与 draft 开销、batch 影响及正确性。MTP 对首 token 的额外影响当前不自动建模；不能据此声称 TTFT 不变。

### C. 饱和吞吐：单独建立资源需求账

**饱和 TPS 不能用 `1/TPOT` 推断。** 在同一输入/输出长度分布、卡数、软件版本与硬件配置的饱和压测中，测得容量 `R₀` 和各资源的忙碌比例 `u_r`。对候选估计**每个输出 token 的资源需求因子** `f_r`（小于 1 为消耗下降）：


$$
\widehat R_{\rm cap}=R_0\,{\max_r u_r\over\max_r(u_rf_r)}.
$$


这是**归一化瓶颈需求代理**，假设硬件能力与 workload 不变；没有提供完整资源因子就**拒绝输出吞吐预测**。低利用率调度、资源竞争、batch 形态或专家负载变化会让它失准，必须用多种并发校验。

若另有请求到达率 `λ` 与平均真实输出长度 `L`：


$$
\widehat R_{\rm served}\le \min(\lambda L,\widehat R_{\rm cap}),\quad
\rho={\lambda L\over\widehat R_{\rm cap}}.
$$


`ρ` 是压力指示，**不是队列时延模型**。接近饱和时不能把低负载 TTFT 推成线上 p99；复杂调度需要 trace 回放和离散事件模拟。

### D. 用实验更新预测

每个 scenario 可附 `observed` 的真实 TTFT / TPOT / 容量。输出自动计算


$$
\mathrm{error}\% = 100\,{\widehat y-y\over y}.
$$


保留**保守 / 中心 / 乐观**三组假设、负结果和独立验证负载。当前不自动训练不可辨识的参数，也不伪造置信区间；有足够真实 profile 和独立 A/B 后，可以在物理模型上拟合残差校正。

## 合成演示（非 H800 实测）

| 候选情景 | 平均 TTFT | 平均 TPOT | 饱和输出 tok/s |
|---|---:|---:|---:|
| baseline | 150.00 ms | 22.00 ms | 1800 |
| QSA 保守 | 148.33 ms | 20.25 ms | 1894.74 |
| QSA 乐观 | 145.94 ms | 18.60 ms | 2195.12 |
| MTP 假设 | 150.00 ms* | 15.87 ms | 1578.95 |
| PLE 重叠 | 146.97 ms | 22.00 ms | 1800 |

\* MTP 对首 token 未单独建模；此处数值恒定是模型范围限制，不是物理结论。

本合成样例到达率 `8 req/s`、平均输出 `128 token`，实际到达需求仅 `1024 output tok/s`。因此容量上升不意味着低负载时的实际吞吐也会跟着上升。MTP 情景更展示出：**单请求 TPOT 改善，但饱和容量可能下降**。

## 从零数据到 H800 实测模型

请实施 Agent 先按 [基线收集任务卡](examples/measurement-checklist.md) 整理：

| 数据 | 用途 |
|---|---|
| checkpoint/revision、引擎 SHA、GPU 数、dtypes、workload hash | 公平性与可复查 |
| 同一批低负载请求的 TTFT / TPOT 均值、prefill / first decode 均值 | 基线与阶段校准 |
| 不重复计费的 block 关键路径及重叠窗口 | 各 trick 可消除时间 |
| 饱和输出 tok/s 与资源忙碌比例 | 吞吐容量预测 |
| 局部提速、额外开销、资源需求变更、质量检查 | A/B 情景及约束 |
| MTP 实际接受长度、验证时间、draft 成本 | MTP 单位最终 token 成本 |
| 盐化前缀键、有序请求、hit token 数、混合 state MiB、驱逐次数 | 业务条件下的 Prefix 命中与容量天花板 |
| 压缩前后 token、压缩耗时、输出质量与新的 prefix key | 压缩后的真实净收益与质量约束 |

路径：**v0.1 结构情景模型（已实现） → v0.2 精确键 LRU trace 回放（已实现） → 真实 H800 校准（待数据） → 分块/多池/并发模拟（待做） → 留出集检验与残差学习（待做）**。

## 参考与边界

- [Amdahl 1967](https://doi.org/10.1145/1465482.1465560)：部分加速的总收益上限。
- [Roofline 原始论文](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)：计算/内存资源限制。
- [Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)：speculative decoding 验证轮产出。
- [Vidur, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html)：profile 驱动的端到端推理模拟。
- 本仓库原文档包含 [16 项一手资料](../../theory/optimization/benefit-prediction.md)。

主预测器**不会预测**未经输入的 prefix 命中率；新的 cache 模块可以对**已给出的按序精确键 trace**做有限容量回放，但仍不能预测未经验证的 p95/p99、排队 TTFT、多节点 P/D 传输、显存溢出退化或动态专家倾斜。正式报告仍要按 [Benchmark 协议](../../docs/benchmark-protocol.md) 测分位数、失败率、质量与 GPU 预算；不允许通过改精度、模型或回答长度冒充优化收益。
