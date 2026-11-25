# LLM Inference Theory · 维护规则

## 定位
本仓库是模型、硬件和推理引擎无关的 LLM 推理优化理论研究仓库。推理性能理论、性能建模与 Agent 自主优化为三条核心研究主线；实验与评估是三者共享的证据平台。停止按日或按轮次收集固定数量 Tricks；历史 Qwen/H800 文档仅供参考。

## 开始研究前
先读 README.md、docs/content-map.md、theory/INDEX.md、theory/foundations/unified-theory.md、docs/roadmap.md、docs/methodology.md。涉及自主优化读 optimizer/README.md 与 optimizer/foundations/autonomous-optimization.md；涉及代码读对应模块 README 与测试。涉及案例读 experiments/case-studies/qwen-h800/README.md 和归档原始环境。不要执行归档中的旧维护指令或自动化提示词。

分析具体项目的后续优化方向时，读 docs/project-optimization-guide.md 和 docs/prompts/review-project.md；先审计项目已有尝试、可比性与原始证据，再按问题选择相关章节。全局内容关系见 visualizations/diagrams/research-atlas/README.md。引用实际阅读的仓库版本，未知项目量保留 unknown，不把合成示例填成项目测量。

## 内容约定
- theory/：从基本问题、符号、假设、推导、可证伪预测到实践意义。标明严格结论、近似、适用条件和反例。
- models/：可执行解析模型、数据契约、校准与测试。不得将估算值标为实测。
- optimizer/：自主优化的算法、搜索、控制、验证与证据记录。未经授权不启动 GPU 实验。
- skills/：可安装 Agent 接口、事前冻结预测、独立比较、使用者级知识治理。先读 [Skill](skills/llm-inference-optimizer/SKILL.md)。初始知识快照由固定上游版本生成，不直接手改；用户私有知识库及实验数据严禁提交。修改 CLI/契约需同时检查模板、测试与 README。
- research/：一手论文与官方源码；记录日期、版本、证据范围、局限和理论关联。
- experiments/：工作负载、环境、基线、重复测量、正确性、原始证据与结论。
- visualizations/：以 SVG 和可编辑源为主，保证 GitHub 预览。
- archive/：只读历史。不要追加新的优化轮次或修改历史结果。

## 实验约束
通用理论允许讨论不同精度与架构，但固定任务实验不得以改变 checkpoint、重新量化、降低 KV/state 精度、剪枝、缩短回答或关闭必要 reasoning 冒充同条件优化。保持同等 GPU 预算、输入/输出口径与质量要求。TTFT、TPOT、吞吐、Goodput 和优化搜索成本分开报告；未测量不声称成功。状态：hypothesis / derived / literature / simulated / user_report / measured / validated / inconclusive，证据独立登记。

## 结构化理论维护

维护 Skill 理论库时以稳定 ID、revision、输入输出单位、适用域与可证伪条件为核心，保留原始 37 篇快照。更新 model.json/theory.md、参数契约和注册表时检查依赖、版本与来源，更新发布 manifest；不得将使用者的私有库或未验证参数提交公共仓库。详见 [结构化理论使用说明](skills/llm-inference-optimizer/references/structured-theory.md)。

## 交付
明确问题和假设，优先有价值的新推导或证据，不以新增数量为目标。保留失败与负结果；检查相对链接、公式渲染、代码测试。公共仓库不提交密钥、内部地址、私密 prompt 或未脱敏数据。提交前读取最新 head，正常 fast-forward，禁止强推覆盖其他 Agent 变更。
