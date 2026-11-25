# 结构化理论库 v2：查询、计算、证据与演化

[六层总索引](../knowledge/structured/INDEX.md) · [理论图谱](../knowledge/structured/ATLAS.md) · [JSON Schema](../knowledge/structured/schemas/README.md) · [原始资料](../knowledge/INDEX.md)

## 1. 两个组织维度

六层是理论依赖层次（定义、规律、性能、优化、Agent、学习）；Seed/User/Project 是所有权和演化范围，两者正交。基础定理、可校准模型、方法/假设分开标记；原始 37 篇资料保留。初始 33 个单元全部有 model.json 和 theory.md。参数库初始为空，不填充假实测。

## 2. 查询和计算

所有命令在 Skill 目录使用 Python 3.10+；如无个人 active，查询命令读取只读 seed，不创建目录。已有旧 v1 库必须先显式迁移，避免将 seed 当作个人现状。

```bash
python3 scripts/skill.py validate-library
python3 scripts/skill.py catalog roofline
python3 scripts/skill.py lookup PERF-OPERATOR
python3 scripts/skill.py impact PERF-OPERATOR
python3 scripts/skill.py calculate FND-ROOFLINE assets/roofline-input.json
```

lookup 返回模型定义、path、先修闭包、下游和已登记参数集；impact 是依赖影响提示，不证明下游数值一定改变。calculate 仅接受白名单中的 16 个窄解析函数。输入必须按字段提供 value/unit，量纲显式，不执行 formula 字符串；输出标记 conditional_calculation_not_measurement。输入单位严格匹配；预测引用 byte/bytes 同义，s 和 ms 必须显式选择或换算。DAG 输入 record 为 nodes=[{id,time_s}] 与 edges=[[from,to]]。耦合 scalar 还需 objective_unit；拟合 record 为 {unit,pairs:[{actual,predicted}]}；误差 record 为数组。算子、通信的参数须显式提供，或给出一个已登记 parameter_set 与匹配 context_id；两者不能混用。Roofline 示例 2 秒只是在给定必要工作和上界下的条件界，并不是 GPU 实测。

## 3. 模块与预测绑定

在 prediction.json 的 metric 中使用：

```json
{"model_ref":{"id":"PERF-OPERATOR","revision":1,"output":"time_ms","parameter_sets":[]}}
```

若无法结构化定位，使用 model_gap 写明原因；它可以用于数值缺乏模型身份的外部预测，但不能冒充模块级证据。每个 v2 metric 二选一。parameter_sets 可指向主模型或其先修中的参数集；context、revision、参数不能冲突。严格界必须匹配物理界输出，近似预测不得冒充界。冻结记录会保留这些字段，旧记录不回填。holdout 的 shadow.model_refs 以指标名为 key，与 shadow.points 一致。CLI 会检查所有目标。

compare 生成 module-review-template.json。每个主模型/指标填写 keep / observe / calibrate / correct / extend / empirical / revise / inconclusive 与非空依据；必要时登记其先修模型和关联指标。保存为轮次根目录 theory-update.json，record 验证并收录。module-summary 按理论版本、context、模型 ID/revision/output、参数集和指标定义分组，报告点误差、区间、界与样本数，另外统计未定位缺口。误差不能单凭主模型身份判因果。

## 4. 模型候选和参数集

参考 [proposal.json](../assets/proposal.json)（仅模板）和 [理论演化](theory-evolution.md)。结构化修改由 structured_changes（相对 structured 的路径到完整 JSON 对象或 Markdown）和 parameter_sets（新增 ID 到参数集对象）表达。每个修改模块同时提供 model.json 与 theory.md，既有 ID/path 不变，revision +1；新模块 revision 1 且无未经实现 evaluator。变化触及原 evaluator 的公式、接口或参数契约时必须禁用旧 evaluator，直到代码与测试独立更新；不能把任意表达式执行当作实现。基础数学/物理定义只能以 revision 类提议并接受理论审查，不能拟合。

参数集例（示意，必须换成真实证据）：

```json
{
  "model_id":"PERF-OPERATOR",
  "revision":1,
  "context_id":"actual-context",
  "scope":"document exact hardware, operator, shape, dtype and workload",
  "method":"document independent fit and identifiability",
  "values":{"exposed_overhead":{"value":0.001,"unit":"s"}},
  "evidence":["actual-project/train-001","actual-project/train-002"]
}
```

参数数值/单位与显式 bounds 匹配；训练引用须是同 active 版本、同 context 的可比 measured search，且链接到该模型或其下游。添加新 ID，不原位覆盖。拟合算法是否可辨识由 reviewer 核查。propose 生成完整候选及 SHA-256，计算直接变更、依赖闭包、过期参数和新增参数；更新过的 revision 的旧参数留在历史版本，候选中退休。registry/dependencies 由模块和有语义的 relations 重建，不能直接编辑生成结果。

独立 holdout 冻结同指标的 active/shadow 数值与各自模型引用，完整登记轮次。promote 既检查原 MAE/留出/批准门槛，又要求每目标各 context 不退步、所有变更模型至少在一条合格的配对预测链中、所有新增参数集在合格 shadow target 实际被引用。数值证据仍不等于证明规律；非数值方法、基础定理修订等需要专门的独立 benchmark 或数学审查，无法满足数值门时保留为候选，不用不相关的预测误差冒充验证。不得伪造 reviewer reference。

## 5. 用户库与显式迁移

默认用户库路径仍为 ~/.local/share/llm-inference-optimizer；init 不覆盖已存在 active。新用户创建 seed-1b4c5a377634-v2。先执行 resolve 检查 schema_version/migration_required。老用户的 v1 版本、原始冻结预测、证据与私有文档继续可读；不要回填过去的预测引用。

对于相同初始源 commit 的旧库，在备份并审查本地文档后，由用户明确批准：

```bash
python3 scripts/skill.py resolve
python3 scripts/skill.py migrate migrated-v2 actual-review-ref
python3 scripts/skill.py validate-library
```

migrate 不覆盖旧版本；复制现有 documents 并附加新结构化 seed，建立有 parent 的新版本。migration.json 记录不同于上游的本地文档列表：结构化初始模型来自上游，不自动反映这些私有改动。必须先核查冲突和适用域，再用于预测；对不同 seed 来源拒绝自动迁移。旧证据仍原样保留；需要时可以通过 rollback 恢复祖先版本（真实批准引用）。迁移只证明布局改变，不是科学验证。用户数据只保存在本地；同一 context 的身份需要由项目定义和审查，而不是仅凭 GPU 名称自动匹配。

## 6. 验收

在 Skill 根目录执行 `python3 -m unittest discover -s tests -v`。测试覆盖旧冻结兼容、版本完整性、依赖循环、单位、16 个解析参考、模块复盘、参数留出和迁移。真实推理服务的通用性能预测器、自动 profiler、残差训练器和完整自动 GPU 优化仍不是本版实现范围。
