# Round 03 · 给实施模型的任务书

研究日期：2026-11-20。所有任务都配合 [optimize-qwen-h800](../../skills/optimize-qwen-h800/SKILL.md) 执行。先读 skill，再读一张卡；每阶段有检查结果再推进，不要求模型自行把十篇研究拼成实施方案。

## 可直接复制的提示词

把下面的 T027 换成选定的一个编号，并填写实际实施代码仓库。若执行工具支持 `$skill-name`，可以显式调用 `$optimize-qwen-h800`；不支持时要求模型先读取同一份 SKILL.md。

```text
请使用研究仓库 skills/optimize-qwen-h800/SKILL.md，执行 Round 03 的 T027。
研究仓库：Sunhuahai/qwen38-next-h800-tricks 的最新 main。
实施代码仓库：填写实际已授权路径。
任务卡：rounds/round-03/tricks/T027-full-logits-batch-sharding.md。

先完成 S0/S1：读取 AGENTS.md、实际环境与现有最佳有效 baseline，
核对卡片的 SHA、代码入口、实际 shape/dtype/backend 和适用条件。
已有日志与结果直接复用，不重跑 T001–T013。
只确定一个变量，写 experiment-plan.json；按 skill 的 validate_plan.py 检查记录。
结构检查通过不等于正确性通过；仍须完成卡片的真实代码/状态/精度审计。

条件成立后按 S2→S5 做最小原型、正确性、单变量 A/B 与交付。
一次只改本卡的一个阶段，不批量实现其他 trick。
固定 checkpoint、全部实际精度/math/scale、采样/输出/业务功能和 GPU 总预算。
不要把官方例子、别的模型、A100/H200/Blackwell 的结果当成用户 H800 结果。
没有 GPU 或相关访问时继续完成可做的源码/CPU分析，明确阻塞的阶段与缺项。
不要通过改 dtype、关闭 Graph/MTP、减少输出、减少专家或改变采样来绕过失败。

每阶段输出：阶段、观察到的事实、检查结果、下一步。
最终交付：唯一变量、最小 diff、实际代码 commit、精度/质量证据、
真实 A/B 数据、机制指标、TTFT/TPOT/吞吐/错误/内存、状态与具体回滚方法。
成功与负结果都写回已有 feedback/results 入口；没有运行就不填性能数字。
```

## 分派时只给模型一个最小目标

| 卡片 | 第一阶段交付 | 通过后才实施 |
|---|---|---|
| T021 | 当前 QSA score/top-k 时间与 tensor 账本、原 tie/mask 合同 | 分页 4-head 的精确融合原型 |
| T022 | selected block 重叠、union 密度、总字节与 gather 成本估计 | 小 query tile 的共享取数 |
| T023 | 单层 FP32/原 math 的树祖先状态 oracle；列出 QSA/PLE 缺口 | 混合模型分支验证，保持 Graph 合同 |
| T024 | 固定 K 的真实 history 命中与 proposal 语义 | MTP/history 组合，保留正确验证 |
| T025 | GateLinear 实际 dtype、路由分派边界与 top-10 合同 | 跨 GEMM→router 边界融合 |
| T026 | shared expert 实际 FC1/activation/FC2 与每道舍入 | 保持中间数值合同的 MLP 融合 |
| T027 | 实际 compute_logits_local 能力、TP/行数、P2P 内存账本 | 完整 logits owner 分片与对照 |
| T028 | 真实 expert-row/tile 分布、当前 scheduler 与 scale 布局 | 原 GEMM 算术下的常驻 tile 工作队列 |
| T029 | 逐层 raw/padded page 与整池总分配账本 | 固定逻辑 block 的物理打包适配 |
| T030 | unique expert trace、原字节/scale、miss 流量、Graph/EP 支持 | 完整 step 所需专家驻留后调用原 kernel |

第一阶段发现“已内建、无瓶颈、无容量收益或支持硬缺口”即给出证据并结束/阻塞该阶段，不强行制造修改。

## 明确交付边界

- **本研究仓库**：任务卡、引用、环境与实验记录；不包含可直接替代 vLLM 的实现。
- **实际推理仓库**：用户指定 Agent 执行代码修改与测试。用户尚未登记的路径和参数保留 unknown，先从现有运行记录读取。
- **结果位置**：使用运行日期所属周的 `feedback/` 与 `results/`；沿用 [统一 benchmark 协议](../../docs/benchmark-protocol.md) 与 [CSV 表头](../../templates/result.csv)。
- **任务生命周期**：T001–T013 为 archived/attempted；本轮是研究候选。编译通过或局部更快都不能直接变成 validated。

R01–R03 仍可使用相同 skill，但派发时只选其一个阶段和对应 T 编号，不把第二轮与本轮一起叠加测收益。
