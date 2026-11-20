# Round 03 · 模型专属研究判断

更新：2026-11-20（Asia/Shanghai）。目标：Qwen3.8-Flash-Next-FP8 / Qwen4Exp / H800。参考源码固定 vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`；[一手来源与 PR head](sources.md)。**本文件是研究判断，未进行 H800 实测。**

## 1. 这轮解决什么

用户已经明确 T001–T013 都尝试过；它们移入 [Round 01 归档](../../archive/round-01/README.md)，不重新派发。T014–T020 与 R01–R03 仍按原研究保留，本轮排重覆盖这些内容，新增恰好十项 T021–T030。

新的候选按实际修改的计算对象或协议区分。每张卡都先列“当前源码已经有什么”，再指出仍缺的跨边融合、跨 query 复用、模型接口、物理布局或状态语义。这样能防止实施模型把现有开关、另一个 GPU 的内核或同一旧机制换个名称再试一次。

同时把执行过程收敛到 [参考 skill](../../skills/optimize-qwen-h800/SKILL.md) 的 S0–S5：先复用真实环境和基线，再判适用性，做小原型，通过正确性后测单变量 A/B，最后写证据与回滚。当前环境清单仍有 unknown；这表示仓库缺记录，不表示用户没做过实验。

## 2. 十个新增点的证据强度与边界

| ID | 本轮新增对象 | 一手依据支持到哪一步 | 还需要证明什么 |
|---|---|---|---|
| T021 | QSA score→精确 Top-k 数据流 | 当前独立 FP32 score 表；FlashTopK 的精确筛选/recovery 机制 | H800、4×128、分页 cache、同分规则与全精度候选移植 |
| T022 | 同一请求的多 query KV tile 读取 | 当前 kernel 是逐 query row；厂商说明跨 query block 复用动机 | 实际 selected 集合重叠、union 开销和固定 mask 下净收益 |
| T023 | 固定节点预算的树祖先与提交语义 | TreeWY 论文/代码提供 GDN ancestor 表示，附宽树负结果 | Qwen4Exp 的 GDN/QSA/PLE 全部状态、固定 math、采样与 Graph |
| T024 | Native MTP 与 history copy 的混合 q | NVIDIA 有同模型 history-drafts 实现，但栈/精度不同 | 当前 H800 栈逐位置真实 proposal、拒绝残差、draft/target 状态 |
| T025 | Router GEMM 到完整路由的执行边界 | 当前 gate 独立、top-k 已融合，准确代码边界可定位 | 同 BF16 logits/FP32 router 合同、dispatch/EPLB 语义与片上资源代价 |
| T026 | BF16 shared expert 的 MLP 内部执行 | 当前三算子与每道 BF16 舍入可定位，已有 overlap 作基线 | 原算术下的三算子融合、FC1 复用、对 routed 分支的资源干扰 |
| T027 | 完整 logits 的 TP owner 分片 | 同模型公开接口补丁与真实 TP tiny 对照 | 本机 runner/功能、真实行数、P2P 内存与服务净结果 |
| T028 | 专家 FC1 的 tile→CTA 调度 | 当前普通 tile grid；CUTLASS 提供 SM90 persistent 机制 | 原 block128 算术下循环、尾 tile/remote skip、实际 occupancy |
| T029 | GDN/PLE 剩余统一尾 padding | 当前统一 padded MambaSpec；通用异构 packing 已有 | 同逻辑 block 下完整 backing pool 的可兑现释放量 |
| T030 | 原字节专家权重/scale 的驻留 | 开放 PR 有逐专家 provider，但并行/Graph 有硬限制 | 本机支持、真实 miss 流量、完整 unique 工作集和不改归约 |

各项的完整代码入口、SHA、分步动作与回滚见 [十卡目录](README.md)。T025/T026 的新融合以及 T022 的局部 union 设计是本仓库基于已读源码提出的 **hypothesis**，不冒充厂商已经完成的 Qwen4Exp/H800 产品实现。

## 3. 三个优先可行性检查

### T027：先补清模型接口，再算通信与容量

当前外层 wrapper 有 `compute_logits_local`，内层 Qwen4ExpForCausalLM 缺同名方法；只查外层 `hasattr` 会误判支持。公开 [PR #61018](https://github.com/vllm-project/vllm/pull/61018) 针对这个具体模型补内层接口。它保留完整 logits，由不同 TP rank 负责不同请求行；不省本地 head GEMM，也不等于 T014 的 sampled-token/argmax 路径。

第一阶段只查实际 runner、方法、TP/采样行数分布和功能门槛。通过后使用同 adapter build、同 NCCL 设置比较开关。公开旧 nightly 的小幅增益混有 channel cap 配置，不能单独归因于分片；另有 [P2P 内存与 KV 容量负证据](https://github.com/vllm-project/vllm/issues/68067)。接口简单不等于服务一定更快。

### T021：score→Top-k 是明确的新数据流边界

当前 QSA decode 的 score 和选择分离。对于足够长的可见块列表，消除完整 FP32 score 表及后续扫描有合理机制依据，但必须计入候选缓冲、筛选、recovery 和展开的全部成本。[当前 QSA 源码](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py)

第一阶段交实际 shape、现有 backend、score/selection 时间和理论可省上限。现成 SM100 融合以及 FlashTopK 的 32-head 连续接口均不能直接接本模型的四头分页路径。精确选中集合、tie、tail 和有效计数是验收对象，不能只测高 IoU。

### T029：先做 CPU 容量账本，能否决就早结束

本模型有不同大小的 GDN 与 TP-replicated PLE 状态；统一 padded page 可能留下空洞。但 QSA 和主 KV 的异构 packing 已存在，不能把它再计一次。[当前 Mamba spec](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/mamba/abstract.py) · [planner](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/v1/core/kv_cache_planning.py)

第一阶段固定逻辑 block、checkpoint 和池策略，仅模拟去掉 GDN/PLE 的多余统一尾部。必须比较完整 pool 的 stride、组数和同 trace 占用；逐层页小了不代表物理 HBM 真能释放。总账无改善就退出，不继续写 allocator。这是较弱实施模型可以按模板完成、也容易明确否决的任务。

## 4. 激进候选的明确停止线

| 候选 | 应先停止的情况 |
|---|---|
| T022 | selected 集合重叠低、原路径已命中 L2，或者 union/mask/额外 FLOPs 超过少读收益 |
| T023 | 只改 GDN，没有 QSA/PLE 祖先状态；把 FP32 state 标签到 BF16 dot 当作固定精度；不能保持采样或 Graph 合同 |
| T024 | 只替换 token，却仍拿旧 MTP logits 当 proposal q；改变实际随机采样为 greedy；接受长度不覆盖查找成本 |
| T025 | 已有融合或无暴露瓶颈；跨越未适配的 dispatch；路由 ID、tie 或 EPLB 映射改变 |
| T026 | 丢失 SiLU 的 BF16 中间舍入；每个 FC2 tile 重算整个 FC1；shared 变快却挤慢 routed 分支 |
| T028 | 实际 backend 已 persistent；工作量不足一波；为改调度同时重排 K/scale 或切换 math |
| T030 | 无显存压力；每步 unique 超缓存容量；miss 流量过大；EP/DP/SP/FULL Graph 不可保持 |

树验证和专家驻留保留为高工程储备。前者完整模型路径当前 `blocked`；后者对当前上游不支持的并行/完整 Graph 组合 `blocked`。这不妨碍完成对应 CPU/单层可行性阶段，也不能因此假装已经支持整模型。

## 5. 参考 skill 的设计决策

Skill 主文件控制执行顺序，三页参考分别提供模型合同、按风险分支的验证和交付字段。模板把“唯一变更”与“必须不变的合同”分开，脚本核对空字段、多变量、归档任务，以及 baseline/candidate 的 dtype、scale、math、Graph、MTP、采样、thinking、负载和资源记录。

它只核记录结构，不验证证据真实性；不会因为 JSON 填满就认定优化成功。实施模型仍要按卡片核源码、实际 tensor/状态与运行结果。每卡最多 6–7 个有顺序的步骤，明确第一阶段应产出什么与何时结束；所有现行第二轮卡也已补 skill 入口。

## 6. 本次交付与验证边界

本次交付：13 卡归档及旧路径跳转、Round 03 十张新卡、源码/来源索引、总任务书、参考 skill、首页/本周/反馈/决策更新。文档检查覆盖 ID 数量、跳转、相对链接、引用、固定版本和状态；skill 检查覆盖 metadata、计划结构与独立只读任务试用。

**没有 GPU benchmark、没有本机加速比、没有已经实现十项的声明。** 新候选的价值要由实际 profile、可否决门槛和单变量结果决定。后续执行按 [benchmark 协议](../../docs/benchmark-protocol.md)记录，实验轮次与日期目录分别维护。
