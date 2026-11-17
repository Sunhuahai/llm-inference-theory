# H800 推理优化综述：三页技术图

**三页高信息密度、可编辑的 draw.io 技术综述**。面向 Qwen3.8-Flash-Next 的 H800 推理优化，统一固定 checkpoint、量化精度和总 GPU 预算；核心指标是 TTFT、TPOT 和饱和输出吞吐。

> **证据边界：** 图中 12 个 trick 是条件候选，尚无用户 H800 上的实测加速结论。第 3 页的数值均为合成演示，不能引用为实验成绩。

**[下载三页 draw.io 源文件](inference-optimization-map.drawio)** · 画布 1920×1080 · 原生 mxCell、文字、连线、表格均可单独编辑。

## 01 · 推理性能分解与瓶颈识别

[查看独立 SVG](previews/01-performance.svg)

![推理性能分解与瓶颈识别](previews/01-performance.svg)

- 请求生命周期：接入/调度 → Prefill → 首 token Decode → 迭代 Decode → 输出。
- 低负载**时延关键路径**：TTFT、TPOT 的时间定义；串行 + 计算 + 通信 − 重叠。
- 高负载**资源容量**：GPU Compute、HBM、互联与 CPU 作为不同瓶颈。
- 公平比较：实际输出长度、缓存冷热、模型 revision、dtype、总 GPU 数量、RPS/并发必须固定或明确分组。

## 02 · 优化策略与架构映射

[查看独立 SVG](previews/02-strategies.svg)

![优化策略与架构映射](previews/02-strategies.svg)

三类策略、12 条独立 trick，在同一张矩阵中对应**机制—指标—实施前提/副作用**：

| 类别 | Trick |
|---|---|
| Prefill / TTFT | T001 分块 Prefill；T005 GR/PLE 序列并行；T008 Prefix 状态缓存；T011 Host/Tokenizer |
| Decode / TPOT | T002 CUDA Graph；T003 QSA prepare；T004 QSA 投影；T006 PLE 预取；T009 原生 MTP；T010 MoE 通信 |
| 多卡与服务 | T007 TP/EP/DP；T012 P/D 分离 |

实施前必须检查实际 build 是否已包含该优化，避免重复归功。具体实验设计与一手来源仍以 [周目录](../../weeks/2026-W47/README.md) 及 trick 卡片为准。

## 03 · 优化收益预测模型与实验校准

[查看独立 SVG](previews/03-prediction.svg)

![优化收益预测模型与实验校准](previews/03-prediction.svg)

从三组实测/假设参数进入模型：

1. **Baseline**：低负载 TTFT/TPOT、Prefill/首步拆分、饱和输出 tok/s。
2. **Profile**：互斥 block、计算通信重叠、资源压力及约束下界。
3. **Intervention**：各 lane 的局部加速、新增开销、MTP 接受与验证成本、资源需求因子。

输出三项指标的**情景预测**及瓶颈变化，并在保守/中心/乐观情景下进行单变量 A/B、带符号误差与留出集验证。它是理论约束 + 基线校准模型，不会凭空产生 H800 实测参数。

可运行的预测器及数据字段说明：

- [模型推导与使用方法](../predictor/README.md)
- [Python CLI](../predictor/predict.py)
- [合成输入和输出](../predictor/examples/synthetic-prediction.md)
- [H800 实测数据收集任务卡](../predictor/examples/measurement-checklist.md)

## 编辑与维护

- 运行 `python3 theory/diagrams/build_diagrams.py` 重新生成同一份三页 draw.io 与对应三张 SVG。Python 标准库，无外部图形依赖。
- `python3 theory/diagrams/validate_diagrams.py` 检查页数、页面尺寸、图元 ID、边界与 SVG 合法性。
- **脚本生成是确定性的**。若在 diagrams.net 手动修改了图，须同步到脚本后再重建，否则生成的三页会覆盖手动布局。
- SVG 是基于同一组几何元素生成的轻量预览；主格式始终是可编辑 draw.io。
- [上一版 11 页图解](https://github.com/Sunhuahai/qwen38-next-h800-tricks/tree/beb5096ec16d8ab008848f7a857402dc36881051/theory/diagrams) 可在 Git 历史中查阅与恢复；当前主线只保留三页，旧预览不再作为现行图解使用。

理论依据与细节见 [完整优化收益预测综述](../optimization-benefit-prediction.md)。
