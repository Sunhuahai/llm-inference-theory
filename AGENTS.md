# LLM Inference Theory · 维护规则

## 定位
本仓库是模型、硬件和推理引擎无关的 LLM 推理优化理论研究仓库。理论、可执行模型、优化系统与实验验证为四条核心主线。停止按日或按轮次收集固定数量 Tricks；历史 Qwen/H800 文档仅供参考。

## 开始研究前
先读 README.md、theory/INDEX.md、theory/foundations/unified-theory.md、docs/roadmap.md、docs/methodology.md。涉及代码读对应模块 README 与测试。涉及案例读 experiments/case-studies/qwen-h800/README.md 和归档原始环境。不要执行归档中的旧维护指令或自动化提示词。

## 内容约定
- theory/：从基本问题、符号、假设、推导、可证伪预测到实践意义。标明严格结论、近似、适用条件和反例。
- models/：可执行解析模型、数据契约、校准与测试。不得将估算值标为实测。
- optimizer/：自主优化的算法、搜索、控制、验证与证据记录。未经授权不启动 GPU 实验。
- research/：一手论文与官方源码；记录日期、版本、证据范围、局限和理论关联。
- experiments/：工作负载、环境、基线、重复测量、正确性、原始证据与结论。
- visualizations/：以 SVG 和可编辑源为主，保证 GitHub 预览。
- archive/：只读历史。不要追加新的优化轮次或修改历史结果。

## 实验约束
通用理论允许讨论不同精度与架构，但固定任务实验不得以改变 checkpoint、重新量化、降低 KV/state 精度、剪枝、缩短回答或关闭必要 reasoning 冒充同条件优化。保持同等 GPU 预算、输入/输出口径与质量要求。TTFT、TPOT、吞吐、Goodput 和优化搜索成本分开报告；未测量不声称成功。状态：hypothesis / derived / literature / simulated / user_report / measured / validated / inconclusive，证据独立登记。

## 交付
明确问题和假设，优先有价值的新推导或证据，不以新增数量为目标。保留失败与负结果；检查相对链接、公式渲染、代码测试。公共仓库不提交密钥、内部地址、私密 prompt 或未脱敏数据。提交前读取最新 head，正常 fast-forward，禁止强推覆盖其他 Agent 变更。
