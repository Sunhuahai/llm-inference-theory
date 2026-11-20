# T021 · QSA 评分与精确 Top-k 流式融合

实施者先读 [optimize-qwen-h800 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行本卡一个阶段。

日期：2026-11-20。状态：**conditional，需要移植**。证据：`literature + hypothesis`；本机收益、实现 commit、质量结果均为 `unknown`，未运行 GPU。

## 唯一新增变量

只替换 **QSA indexer 的 score → Top-k 中间数据流**：评分后在设备端保留候选、完成精确选择，避免完整 FP32 logits 表写回再读取。保持原评分表达式、四头归约、512 个压缩块、分页可见范围、选中集合及尾部 token 语义。先做 decode，prefill 是单独后续阶段。

本模型 indexer 为 **4 Q heads × 128、1 K head、压缩比 4**，2048 token 预算对应 512 块。T003 改 norm/RoPE/cache prepare；T004 改投影 GEMM；本卡在两者之后。仅切 cooperative/persistent Top-k、给已有 prefill 分块限工作区、复用 MTP step 0 的 indices，都不算本卡的新实现。T001–T013 已尝试是 `user_report`；T014–T020/R01–R03 结果仍未知。

## 固定版本与实际缺口

源码统一固定 **vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`**，不要求用户升级。下列路径都相对于该源码根目录。

| 入口 | 已核实事实与实施边界 |
|---|---|
| `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py::{qsa_select_paged_decode,_qsa_mqa_paged_uniform_kernel,_topk}` | decode 分配 `[query_rows, page_table_width × page_size]` FP32 logits，再单独 Top-k。这里是首版替换边。 |
| 同文件 `qsa_select_paged_prefill` | 已按工作区上限分块；不要把现成内存保护重新包装为新 trick。 |
| `vllm/models/qwen4_exp/nvidia/qsa_indexer.py::QSAIndexer`、`expand_qsa_block_indices` | 保留完整块到逻辑 token 的展开、0–3 token 尾部；packed 输出末列是有效计数，不能当 token ID。 |
| `vllm/model_executor/kernels/attention/dsa/fused_mqa_topk.py::use_fused_mqa_topk` | 现成融合仅 SM100、32×128 indexer；不适用 H800 的 4×128。其候选编码含 fp24，不能直接当作固定 FP32 中间精度实现。 |

[当前 QSA 源码](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py) · [现成融合限制](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/kernels/attention/dsa/fused_mqa_topk.py)。

一手算法参考是 [FlashTopK](https://github.com/Tencent-Hunyuan/FlashTopK/blob/090902abad86e5bb3324519124a680122fff91a1/docs/api.md)，固定 `090902abad86e5bb3324519124a680122fff91a1`：阈值筛选失败必须 GPU recovery，最终为精确 Top-k；但当前 API 固定 32×128、连续 KV，QSA/分页不能直接调用。只借候选筛选与精确重扫流程，首版不采用 key pruning，不复制任何降低中间精度的编码。作者按“同分取较小 index”打破平局，不能假定这与本机 Top-k 相同。

## 实施步骤（最多七步）

1. **锁基线。** 从已有记录取得 W1 当前最佳有效配置，保留 MTP、Graph、拓扑、cache dtype 和所有 math mode。记录实际源码与上述入口差异；若已经有同等 QSA 融合，标 `baseline-covered` 停止。换引擎/build 必须先建等价基线，不能计入本卡收益。
2. **先给可否决 profile。** 分开统计 scoring、Top-k、logits 分配/读写，以及它们在完整请求中的占比。取真实 query 数、可见块数、页大小和当前 Top-k backend。若消除这段的理论上限也低于基线波动，暂缓；不以随机长矩阵代替业务证据。
3. **先交纯语义小参考。** 保存一小份脱敏形状/合成边界输入；覆盖不足 512 块、恰好 512、长窗口、不同 page 映射、3/4/5-token 边界、同分/近同分。参考输出包含 FP32 scores、block IDs、展开 IDs、valid counts。只做测试可用 CPU 参考，不导出用户原文。
4. **写一个新后端。** 保持当前 QK 运算与 FP32 score/候选值，只把“落完整表”换成有容量界的候选缓冲与精确 rescan。候选不足、溢出、边界同分或非有限值无法保证一致时，逐行回退原路径；不能截掉候选继续运行。新增 dispatch 默认关闭，名字由实现者拟定并标为项目私有。
5. **过正确性再计时。** 比较每行选中集合、尾部和 padding；边界同分必须按基线规则重现，无法重现就回退。仅 IoU > 0.999 不合格。继续比较 attention 输出、目标 logits 和固定质量集合；确认没有悄悄换 indexer dtype、FP32 accumulation、TF32 设置或 scale。
6. **同 build A/B。** A 为原路径，B 仅启用新后端；固定原 MTP 深度。先 kernel 全调用计时，包含分配、recovery 与展开，再按[测量协议](../../../docs/benchmark-protocol.md)交错至少三次真实服务请求，含长上下文主场景和短上下文回归。单独报告 fallback/recovery 频率；prefill 移植另开单变量阶段。
7. **交付或退出。** 提交实现 diff、manifest、最小回归、各次 CSV、trace 和结论。只有端到端收益超出波动且尾延迟/质量通过才保留；否则 `inconclusive/regressed`。关闭私有 dispatch 即回到原 score + Top-k，不能保留新阈值跨请求状态。

## 收益预测与观测

令 R 为 query 行数、C 为该调用的 logits 列数。完整表占 `4RC` 字节，消除一次写、一次读的名义流量为 `8RC`；这不是实测 HBM 节省，L2 命中、Top-k 多遍扫描、候选写入和重扫都会改变它。估算 `Δt ≈ 原 logits/Top-k 时间 − 候选筛选/重扫新增时间`。若该段占基线请求时间 f、局部加速 s，理想总加速上限为 `1 / (1-f+f/s)`，只作筛选。

必须观测：实际 DRAM/L2 字节、scratch 峰值、候选/重扫行比例、kernel 时间、TTFT/TPOT p50/p95/p99、输出吞吐、错误率、实际输出长度。所有当前数值留 `unknown`；FlashTopK 的其他模型数字不外推到 Qwen3.8/H800。
