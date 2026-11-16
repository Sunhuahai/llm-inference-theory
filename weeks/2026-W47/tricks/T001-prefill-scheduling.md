# T001 — 分块预填充与批调度

- ID：T001
- 状态：researched
- 调研日期：2026-11-16
- 验证：待 H800 实测

## 问题与机制

长 prompt 的 prefill 混入 decode 时，可能抬高逐 token 延迟。分块把 prefill 拆入多次调度，但 chunk 太小又会拖长首 token 等待。vLLM 官方说明 token budget 的大小会形成 TTFT 与 ITL 权衡；该规律是候选依据，不是本模型的收益承诺。[官方调优文档](https://docs.vllm.ai/en/stable/configuration/optimization/)

## 模型与版本门槛

固定 `Qwen3.8-Flash-Next-FP8` 的 revision、全部 dtype、模板与采样。[官方 vLLM 配方](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) 标注最低 0.30.0，[SGLang 配方](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx) 要求至少 0.5.22；这不证明所有优化组合可用。仍需查实际 commit 的 QSA/Gated DeltaNet 分块与对齐限制及启动日志。

## 实施 Agent 步骤

1. 从可运行基线提取 resolved configuration；确认分块是否已默认生效。
2. vLLM 只改 `--max-num-batched-tokens`；SGLang 只改 `--chunked-prefill-size`。以 2048/4096/8192/16384 为初始候选，剔除不合法或超内存点；这些数字是实验网格。
3. 第一轮固定并发上限。保留两个 Pareto 点后，另开实验扫并发，避免一次改变两个因素。[SGLang 调优](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)

## 单变量 A/B

A 为原 token budget，B 每次只换一个值。使用短入长出、长入短出、长短混合三组相同请求及到达轨迹；固定缓存模式、输出约束和 GPU 数。报告 TTFT/每请求 TPOT/ITL 的 p50、p95、p99，输出 tok/s、请求成功率、排队与抢占次数。先比同负载延迟，再比较同 SLO 的最高吞吐。

## 正确性与回滚

核验跨 chunk 的递归状态、长 prompt 输出与基线一致性；异常、OOM 或明显回归即恢复原参数。质量检查采用固定测试集与约定容差，不以浮点逐位一致作为所有 backend 的通用条件。

## 已知与未知

已知：分块是调度权衡。未知：当前 H800 卡数、框架版本、工作负载及最优 chunk；没有本项目加速比。
