# T007 — 按容量与拓扑选择并行布局

- ID：T007
- 状态：conditional
- 调研日期：2026-11-16
- 验证：待 H800 实测

## 问题与机制

扩大 TP 会改变计算粒度和 collective 成本；减小 TP 再增加副本会复制权重与缓存。EP 又会改变专家分片方式，不能把三者简单等同。vLLM 文档给出 EP 与 TP/DP 的组合语义；H800 实际链路应按具体板卡和机器读取。[并行文档](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) · [HGX H800 拓扑](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html)

## 模型与版本门槛

先读取卡数、显存、节点数与分片维度。[官方 vLLM 配方](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) 标注最低 0.30.0，并指出官方 FP8 的 128-wide block 与普通 TP8 不兼容；有 8 卡且其他条件满足时才考虑该配方的 TP8+EP。PLE 初始不支持 PP；80GB Hopper 示例还依赖 PLE CPU offload。固定全部 dtype、offload 设置及实际 commit，不把 H100 配方当 H800 实测。

## 实施 Agent 步骤

1. 记录 `nvidia-smi topo -m`、P2P 状态与 GPU 可见顺序，按实际链路放置 ranks；用 `nvbandwidth` 核对可达带宽。[NVIDIA 检查方法](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/performance_and_tuning.html)
2. 算清权重、递归状态、KV、graph 与 activation 总预算，生成合法且可容纳的布局；不预设用户拥有 8 卡。
3. 同卡数比较可行 TP 布局；多副本和 attention-DP+EP 作为独立实验，后者须有该模型实现证据。

## 单变量 A/B

每轮仅改变一种布局因素；固定总 GPU 数、相同到达 trace、输入输出与缓存模式。报告单请求 TTFT/TPOT、服务总输出 tok/s、tok/s/GPU、p95/p99、每 rank 显存、collective 占比及排队。不能拿增加 GPU 数的结果当纯算法提升。

## 正确性与回滚

先做分片加载、短长请求与输出质量检查，再压测；不合法维度直接标记 blocked。恢复基线 GPU 映射与并行参数即可回滚，不为实验修改 BIOS 或宿主机隔离配置。

## 已知与未知

已知：最小可行分片不必是最快布局。未知：用户硬件、完整内存预算及 QSA/MoE 后端的合法并行集合；不外推 H100/H200/B300 数字。
