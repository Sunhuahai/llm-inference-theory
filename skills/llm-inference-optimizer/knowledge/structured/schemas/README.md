# 结构化理论数据契约（v2）

- model.json：稳定 id、revision（正整数）、layer、kind、title、formula（仅说明）、inputs / outputs（id、unit、description）、parameters（同样字段 + learnable + bounds）、depends_on、tags、applicability、limitations、validation、sources、evaluator、status、evidence_type。来源路径相对于原始文档根。每模块另有 theory.md。修改结构必须提升 revision；不允许复用 ID 移到其他路径。
- registry.json：ID → model.json 相对路径，由模型文件重建；不接受手工维护的第二份事实源。
- relations.json：人工标注非先修的 feeds/constrains/extends 关系及理由，目标必须存在；dependencies.json 由模型依赖与该文件生成。仅 depends_on 检查无环。
- parameters.json：ID → {model_id, revision, context_id, scope, values, evidence, method}。values 为参数名 → {value, unit}，必须满足模型参数范围，证据为真实轮次 ID；初始集合为空，禁止把合成例或未知量填成校准值。版本内参数集不可原位替换；更新创建新 ID。参数集只对匹配 context 和 revision 有效。
- 预测中的 model_ref：{id, revision, output, parameter_sets?}，绑定 primary output；参数集可以来自主模型或其传递先修，必须匹配 context 和各自 revision。不能结构化定位时填写 model_gap（明确原因），不要伪造。旧冻结记录无这些字段仍保持可读；结构化版本的新冻结必须填写其中之一。
- 单位采取显式字符串匹配（仅 bytes/byte 作为同义），秒与毫秒必须显式换算。record 类型的内部 schema 由 evaluator 文档规定；scalar 的量纲依其明确目标而定，不能用于任意数值互换。数值函数不执行 formula 字符串。
- 证据与模型引用只能表明所用预测链，不能自动断言某个模块导致了偏差；学习、反例与批准由独立实测和审查提供。

验证器：scripts/theory_library.py（通过 skill.py 调用）。初始库无用户参数和实验结论。具体 CLI 输入样例见 Skill 的 references/structured-theory.md。

## JSON Schema

[模型](model.schema.json) · [参数集](parameter-set.schema.json) · [预测引用](model-ref.schema.json) · [模块复盘](module-update.schema.json)。Schema 用于编辑器和交换格式；CLI 的标准库校验另外检查依赖无环、引用存在、版本、单位、证据和参数范围等跨文件约束，不需要安装 jsonschema。
