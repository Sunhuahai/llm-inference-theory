# T022 · 同一请求内多 query 共享 QSA 选中块读取

实施者先读 [optimize-qwen-h800 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行本卡一个阶段。

日期：2026-11-20。状态：**conditional，先测集合重叠**。证据：`literature + hypothesis`；未实现、未运行 GPU，收益与质量结果 `unknown`。

## 唯一新增变量及排重

对同一请求、同一层的相邻 2 个 query，把它们**原本已经选中的** QSA block 组成局部 union；同一 KV tile 只加载一次，再按各 query 自己的精确 mask 计算。目标是跨 query 重用，而非跨 head 的 GQA 重用。第一阶段只处理有两个以上 query 的 prefill 或既有固定深度 MTP 验证；普通单 token decode 原路返回。

Qwen3.8 的 QSA 是 24 Q/2 KV heads、head_dim 256、4-token blocks、512 个完整块加不完整尾部；TP 分片和 KV 复制以本机 tensor 为准。**不改 4-token 粒度、不合并不同 query 的可见集合、不减选中预算、不共享上一时刻 indices。** T003/T004 的 prepare/投影、T021 的 scorer/Top-k 和本卡均分别 A/B。

## 当前代码已有什么

固定 **vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`**：

| 路径/符号 | 证据与改动位置 |
|---|---|
| `vllm/models/qwen4_exp/nvidia/qsa.py::Qwen4ExpQSAImpl.attend` | attention 调用入口；沿用 indexer 输出及 cache scale。 |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py::_qsa_sparse_paged_gqa_splitk_kernel` | grid 按单 query row、KV head、split 展开；一个 CTA 内已复用该 row 的 GQA heads，未在此 H800 路径合并多 query row。 |
| 同文件 `_select_sm90_config`、`_select_config` | 已有 SM90 配置，标注 H20 调优；调 split 数或“开启 GQA”不是新点。 |
| 同文件 `_qsa_merge_splitk_kernel`、`qsa_sparse_paged_attention` | 已有 FP32 partial/LSE merge 与输出 gate；保持这些语义及真实 resolved 路径。 |

[固定 attention 源码](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/nvidia/ops/qsa.py)。[NVIDIA 2026-11-17 技术文章](https://developer.nvidia.com/blog/co-designing-sparse-attention-for-efficient-long-context-inference/)解释了 block 取数在 query 间复用的动机；由此提出 H800 端的局部 union 方案是本项目推导，并非文章已经验证本方案。文章的 FP16 epilogue/accumulate 建议不采用。

[cuDNN QSA 文档](https://docs.nvidia.com/cudnn/operations/sparse_attention/qsa)（检索 2026-11-20）给出精确 mask、dense/gathered 路径，但公开实现面向 SM100/103/107 的训练接口，不是可直接切换的 H800 paged inference backend。不能换成它后宣称完成移植。

## 先算能否值得做

同一 request 内的 m 个 query 选中完整块集合记为 `S_i`；`U = union(S_i)`，`D = sum |S_i|`，复用比例 `r = 1-|U|/D`，union 计算密度 `d = D/(m|U|)`。首版 m=2，不扩到不同请求。

设每块实际 KV 字节为 `B = 4 × Hkv_local × 256 × (bytes_K+bytes_V)`，名义少读上限为 `(D-|U|)B`，**不含 tail/metadata，且未扣原路径 L2 命中**。union 若让每个 query 都算所有 union 项，额外 masked 计算因子约为 `1/d`。这些式子必须用实际选中集合与 cache dtype计算，不能从“文本相邻”猜高重叠。

进入开发前给出 r/d 分布及有效样本比例，结合实测带宽/算力估计：`少读字节 / 有效带宽 > union建表 + mask处理 + 多算QK/PV + occupancy损失`，且预计收益超出波动。若多数 m=2 的集合近乎不交、原 KV 读取已命中 L2、或 union 工作区挤占原 KV 池，标 `parked`。门槛在看到 B 的成绩前确定。

## 七步实施合同

1. **锁基线和调用。** 复用 W1-best、真实 MTP 深度、Graph、dtype/math、TP 组和缓存预算。实际若走别的 QSA backend，先重新定位；若已有跨 query tile 复用，标 `baseline-covered`。
2. **只加采样 trace。** 在 attention 入口采集脱敏 selected IDs、request 边界、query 位置、valid counts，计算上面的 r/d、尾部比例与字节上限；计时正式跑时关闭 trace。交付可复算的小报告，通过门槛才做下一步。
3. **先实现精确 union 参考。** 处理两个 query 的分块列表，输出 union、每行 mask 与原顺序映射。允许共享取数，禁止共享 softmax 分母。末列 valid count、短尾、重复 ID、分页边界分别检查；无资格形状回退，不复制完整 KV 到新池。
4. **新增一个 H800 kernel 分支。** 保留原 K/V bytes、scale 和转换路径，包括原 FP8 V 转换行为；QK/PV、partial output、LSE 的 dtype/math mode 不变。固定容量工作区，设备端建 metadata。先只换读取布局，不同时改 split 配置、融合投影或启用 T021。实际 HiSparse/物理 index 合同尚未支持时停止该分支，不能为实验关闭基线功能。
5. **tiny correctness。** 相同/不交/部分交集合、0–3-token tail、两请求交错、非连续页、Graph padding；每行允许访问的 token ID 必须完全一致。对照原 kernel 输出/LSE/gate 和目标 logits；保留各行原 key 顺序，若实现重排了 FP32 归约须单列误差，不能称逐位等价。质量门槛按既有协议预先固定。
6. **A/B 包含所有代价。** A 原 kernel，B 仅 union 分支，固定原 MTP。计入建表、mask、fallback、额外缓冲与 Graph 开销；至少三次交错服务端回放，主测原有长 prefill/verify 负载，单 query decode 作回归。检查实际 DRAM 下降是否被低 occupancy 抵消。
7. **交付并回滚。** 保存实现 commit、manifest、r/d trace 摘要、全调用 kernel 时间与端到端 CSV。只有真实服务收益超过波动且质量/尾延迟通过才保留。私有默认关闭 dispatch 可立即回退单 row 路径，并释放 union scratch；不触碰已提交 KV/state。

必须报告 union 命中资格比例、额外 masked FLOPs、实际 DRAM/L2 字节、寄存器/spill、scratch、TTFT/TPOT/输出吞吐与错误率。局部理想少读比例不是服务加速比，本卡所有收益均待测。
