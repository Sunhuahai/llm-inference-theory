# T023 · TreeWY 分支验证：固定节点预算的原生 MTP 树

实施者先读 [optimize-qwen-h800 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行本卡一个阶段。

日期：2026-11-20。状态：**整模型整合 blocked；单层语义原型 conditional，高工程储备**。证据：`literature + hypothesis`；无本机 GPU 结果。本卡没有可直接启用的 vLLM 命令。

## 新点与不做什么

用现有 MTP 权重生成有兄弟节点的 draft tree，目标模型一次验证各节点；GDN 通过 ancestor mask 的 WY 表示计算输出，只发布接受路径的状态。最终服务 A/B 的唯一变量是**相同总节点预算下，链形或有分支的拓扑**。不增加训练头、不改 checkpoint、精度、GPU 数或采样；树形并不自动更快。

T009 已试原生 MTP 的静态链长；T015 改验证长度；T017/R01 改线性历史的状态读写。本卡的独立缺口是“兄弟分叉、父节点身份及接受路径”，不能把 ReplaySSM 的计数游标直接当树提交。先完成链形的固定精度对照基础，再比较分支；树拓扑与状态表示不能同时替换后合报收益。

## 一手依据与反证

[TreeWY v2](https://arxiv.org/abs/2608.20961v2)（2026-10-22）研究 Qwen3.5 35B/397B，**不是 Qwen4Exp 验证**。[作者 RFC #54080](https://github.com/vllm-project/vllm/issues/54080)明确：链形下 ReplaySSM 更快；扩树提高接受长度但当时吞吐下降，主要受非因果 mask 导致 Graph 退化影响。该负结果必须保留，不用论文的状态节省数字推导本机加速。

参考实现固定作者分支 **`b073ed6cacfa1cf5ae23111854729e662a72a6d1`**：

| 源码 | 可借用部分与限制 |
|---|---|
| `vllm/third_party/flash_linear_attention/ops/tree_wy_triton.py::tree_wy_tree_commit_capture_triton` | 输入显式 ancestor/keep mask，区别链形 index comparison。**默认 `dot_bf16=True` 会把 FP32 state/中间量转 BF16 做 dot**；state 标 FP32 不足以满足本项目精度合同。 |
| `vllm/v1/spec_decode/tree_spec.py::{TreeTopology,build_verify_attn_mask,tree_sampled_accept,tree_sampled_verify_batched}` | parent/depth、祖先 mask、采样参考。随机参考要求每个 parent 下的候选按实际 q 独立有放回抽样；不能把 top-k 无放回候选套进同一个证明。 |
| `vllm/v1/spec_decode/tree_draft.py`、`vllm/v1/worker/gpu_model_runner.py` | 分支 positions、mask 和 runner 接线参考，不能整包替换当前 runner。 |

[固定 TreeWY kernel](https://github.com/sneha5gsm/vllm/blob/b073ed6cacfa1cf5ae23111854729e662a72a6d1/vllm/third_party/flash_linear_attention/ops/tree_wy_triton.py) · [采样合同](https://github.com/sneha5gsm/vllm/blob/b073ed6cacfa1cf5ae23111854729e662a72a6d1/vllm/v1/spec_decode/tree_spec.py)。

目标入口固定 **vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`**：`vllm/models/qwen4_exp/nvidia/model.py::Qwen4ExpDecoderLayer` 使用 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py::QwenGatedDeltaNetAttention`；原生 draft 在 `qwen4_exp/nvidia/mtp.py::Qwen4ExpMultiTokenPredictor`。主模型另有 12 层 QSA 和 PLE，**不能只改 GDN 就接整模型**。QSA `nvidia/{qsa,qsa_indexer}.py` 的分页、四 token 压缩组和尾部须按祖先路径解释；`nvidia/ple_layer.py::Qwen4ExpPLELayer` 及 `ngram_embedding.py` 的 n-gram/short-conv 历史也必须各分支独立。MTP 接受的 HC residual 是多流 10240 宽，不能拿最终 2560 宽 sampling hidden 替代。

## 七步、每步可停止

1. **能力表与预算。** 复用 W1-best，记真实 runner、固定 MTP、所有 dtype/math、Graph 模式。列 GDN、QSA/indexer、PLE、draft cache、采样器的 parent/depth/accepted-node 支持。当前缺任一项，整模型保持 `blocked`；允许继续只读或单层原型。
2. **固定精度 GDN 小原型。** A 为沿每个根到节点路径逐 token 执行当前 GDN；B 为相同输入的 WY/ancestor 计算。先两兄弟、四节点小树及链退化情形。核 state、归一化 key、pseudo-value、gate、stash、dot operands/accumulator 全部实际 dtype；不能仅关 `dot_bf16` 就假定合同满足，也不能借 FP16 corrections 或更宽松 math mode。误差/状态不通过就停止。
3. **补模型语义，不测加速。** 用显式 parent、depth 和接受 leaf/row 身份，逐项实现 QSA 压缩组/尾部、KV slot、PLE n-gram/conv 的祖先隔离。两个同深度兄弟必须来自不同私有上下文；position 是根位置加深度，不能用 DFS 数组下标。每种状态与独立串行路径比较，未完成不可进入服务。
4. **提交和回滚边界。** 接受 path 映射为有序 token/state 记录，再交已有生命周期；不能把接受节点总数当连续前缀。覆盖首节点拒绝、非左分支接受、EOS 中途、取消、preemption、slot 复用、prefix snapshot。拒绝兄弟不得泄入下一轮，发布 checkpoint 前必须物化正确接受状态。
5. **采样与 Graph 两个独立检查。** 沿用实际目标采样；只有原基线本来 greedy 才可作 greedy 服务实验。随机路径先验证 child proposal q、残差归一化、重复候选和约束采样，CPU 小词表枚举/统计对照后才接 GPU。未证明则随机路径 `blocked`。另外确认固定拓扑 mask/metadata 可 capture，不能静默从 FULL 降为 PIECEWISE 后忽略代价。
6. **等节点 A/B。** 同 build、同固定精度状态实现、同 Graph 合同和固定节点总数 N，A 为 N 节点链，B 为一棵预登记的小树；只改拓扑。对外另报 B 相对 W1 当前最佳链式 MTP 的净结果。先测完整 round，再按协议至少三次交错真实服务回放；不能按更高接受率直接判赢。
7. **交付或退出。** 提供逐层能力表、拓扑、采样证明/反例、状态回归、原始结果与回滚 commit。显存、round 时间或 Graph 退化抵消接受收益即 `regressed/parked`。关闭项目私有 tree dispatch，释放 branch metadata，回到现有链路径；不得将未支持分支伪装为链继续提交。

## 判定公式与观测

每轮实际输出 token 数记 E、完整轮耗时记 C。相对 W1-best 的净加速为 `S = (E_tree / C_tree) / (E_best / C_best)`；N 个节点只有接受路径贡献 E，兄弟节点计算仍计入 C。无接受率上升就必然提速的推论。

报告每轮节点数、接受路径长度、draft/target/commit/mask 各耗时、Graph 命中、state/stash 峰值、完整 TTFT/TPOT、输出吞吐、错误率和质量。36 层 GDN 的 FP32 recurrent payload 全 TP 聚合为 108 MiB/request，仅作状态账本；本卡不重复计算 T017 的链式省存收益。所有本机测量目前 `unknown`。
