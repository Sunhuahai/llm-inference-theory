# 执行与验证分支

选择与卡片有关的一条主分支；涉及状态或分布式边界时追加对应检查。先完成最小正确性，再进入完整服务路径。

## A. 算子融合或 tile 调度

1. 从实际 profile/日志取 shape、stride、dtype、backend、stream。保留原输出作 oracle。
2. 找到函数入口和所有调用者。只在卡片支持的 shape 上分派实验路径，其他 shape 保留原路径。
3. 先做最小形状与尾部形状：空/单 token、非整 tile、少量 active experts 或真实长列数。不要把随机标准矩阵当成唯一数据。
4. 比较输出以及离散中间结果。选择/top-k 要查 ties、无效位置、padding；FP8 还要查原 scale 和 cast 顺序。
5. 再检查目标服务使用的 Graph、padding、并发 stream 与 workspace 生命周期。原型不支持 Graph 时不能宣称服务可采用。
6. 机制计时包含额外 pack/merge/dispatch/probe；端到端按研究仓库协议另测。

## B. 分布式协议或 logits 分片

1. 画出每个张量的原 owner、目标 owner、全局 shape、local shape、进程组与归约位置。
2. 固定 TP/EP/DP/PP 与总 GPU 数，不将更换并行布局混入内核实验。
3. 检查不整除 batch、空 rank、尾词表 padding、多请求采样参数、logprobs 和 MTP 验证调用者。
4. 检查 Graph 捕获、P2P 缓冲、高水位显存、通信顺序和退出/失败路径；不能只比较传输字节理论式。
5. 记录实际传输量、等待、峰值 HBM 与端到端指标；GPU 内存下降但 NCCL 缓冲上升时算总账。

## C. 页布局、缓存或专家驻留

1. 先导出实际 tensor shape/stride、逻辑容量、物理分配、高水位与流量。现有实现已消除浪费时直接记录 baseline-covered。
2. 保持原 prefix/cache 容量策略、专家归属和路由，改变的仅是卡片指定的物理布局/驻留方式。
3. 逐项检查命中、未命中、淘汰、slot 复用、并发读写、Graph 地址稳定性与 fallback。
4. 缓存/ offload 必须等到所需完整原权重和 scale 到位后再计算；禁止超时后跳过低命中专家。
5. 用真实命中/路由 trace 计算流量与收益门槛。随机热分布只能标合成测试，不能代替用户 workload。
6. 报总 HBM、host RAM、PCIe/NVLink 流量、关键路径等待、尾延迟；不能只报告某一个 pool 省了多少。

## D. 混合状态与投机解码

1. 先固定 K、确定 proposal 来源与 verifier 合同。业务本来是随机采样时，不能改成 greedy 作为可采用结果。
2. 在一个请求、一个短分支上比对每步 token、position、接受长度、GDN、QSA KV/indexer/压缩状态、PLE n-gram/conv。
3. 构造全拒绝、只接受部分、全接受、EOS、上下文边界、MTP draft/target 隔离等会触发不同提交路径的用例。
4. 再覆盖 prefill→decode、prefix 分叉、抢占恢复、取消、slot 复用和 padded Graph batch；只运行与当前机制有关的情况。
5. 对树或混合 proposer 逐分支核对父节点、祖先 mask、位置和状态。随机采样必须证明正确 proposal q、accept/reject 与 residual distribution；否则该路径 blocked。
6. 只在这些检查全部通过后比较速度。报告每轮 verifier/drafter 成本、实际提交 token、有效接受率、Graph 覆盖和端到端 TPOT，不只报“draft 多了”。

## E. 最小性能结论

读取 [benchmark-protocol.md](../../../docs/benchmark-protocol.md)。复用 baseline 负载和环境，至少三组交错 A/B；至少记录 TTFT/TPOT 的 p50/p95/p99、输出与总 tokens/s、实际输出长度、错误/超时/OOM、质量和各 rank 峰值显存。

把局部机制改善、端到端改善、原始噪声分别陈述。SLO 未知时给延迟/吞吐权衡，不编一个业务阈值。没有 GPU 结果就写未实测，不填写占位性能数字。
