# T024 · 固定深度原生 MTP 加历史续写候选

实施者先读 [optimize-qwen-h800 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行本卡一个阶段。

日期：2026-11-20。状态：**conditional；混合采样合同未实现/验证前为 blocked**。证据：`literature + hypothesis`；无 H800 实测，收益、正确性和实施 commit `unknown`。

## 唯一变量与排重

在 W1 当前最佳固定深度 MTP 的**同一条 K-token 草稿链**内，用已提交生成历史中的精确 suffix continuation 替换若干候选，其余位置仍由原生 MTP 提议。K、目标 verifier、目标采样、state 写回、Graph 和全部精度均固定。首版仍执行原 MTP forward，将替换 token 喂给下一 draft step；假设收益来自更长的接受前缀，不预支“跳过 drafter”收益。

纯 prompt lookup、ngram 或 suffix proposer 已有实现，不算新 trick。本卡新增的是 **native MTP 与 history copy 的逐位置组合及实际提议分布 q 的接线**。区别 T009 的链深度、T015 的自适应长度、T017/R01 的 GDN 写回；不训练新 draft 权重，不改上下文或输出。

## 已读的一手代码

固定 **vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`**：

| 路径/函数 | 当前事实与需要新增的位置 |
|---|---|
| `vllm/v1/worker/gpu/spec_decode/suffix/speculator.py::SuffixSpeculator.propose` | 已有 GPU 局部 suffix 与跨请求已完成输出 corpus；哈希候选仍做精确 token 验证。不能把这项现成能力当新发现。 |
| `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py::MTPSpeculator`、`target_dependent_ar/speculator.py::{propose,_prefill,_decode}` | 原生 MTP 使用自身 proposal 链；需要在每个 sampled token 写入下一步输入之前插入 copy mask/id，不能整条链生成后再拼接。 |
| `vllm/models/qwen4_exp/nvidia/mtp.py::Qwen4ExpMultiTokenPredictor` | 保留多流 target residual、实际 embedding 与 draft QSA cache 接线。现有后续步 `set_skip_topk` 不属于本卡新增点。 |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py::RejectionSampler._verify`、`rejection_sampler_utils.py`、`draft_logits.py::DraftLogitsCache` | 当前接收 draft logits；混合 copied/MTP rows 要显式表达各自 q。绝不能只换 token 而留下旧 MTP logits 当提议分布。 |

[MTP 链入口](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py) · [验证器](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py)。

新增模型专属先例来自 NVIDIA [hyperflare MTP 设计](https://github.com/NVIDIA/hyperflare-inference/blob/6555f9397bd7f39b02b836b7f51ce050a54f26a6/docs/mtp-design.md)，固定 **`6555f9397bd7f39b02b836b7f51ce050a54f26a6`**。其 Qwen3.8 图支持 history copy；`ignition/recipes/qwen38/speculative_model.py::PreparedSpeculativeRound.run` 先跑 MTP，再以 copy mask 替换 token，并将 q 改为点质量；`helios/src/scheduler/decoded_history.rs` 维护有序已提交历史。**该运行时是 SM120/121，示例 checkpoint 为 FP4-Local，本项目不安装/切换它，只移植机制。** 原生 H800/FP8 收益未知。

## 正确性合同先于性能

复制决策须只依赖该位置之前的确定历史/既有规则，不能在看到本位置 MTP 随机抽样结果后再条件选择却仍声称 q 是点质量。若第 j 位确定复制 token c，则 `q_j(c)=1`，其余为 0；MTP 位必须保留实际生成该候选的 q。随机拒绝采样接受率为 `min(1,p_j(x)/q_j(x))`，拒绝后从归一化 `max(p_j-q_j,0)` 取修正 token。copy 位接受概率就是 `p_j(c)`，可能比 MTP 更差。grammar、惩罚和 temperature/top-p 等仍属于实际目标分布 p，不随历史命中取消。

只比较字符串相同不构成随机分布证明。小词表检查混合 q 与残差，目标有零概率、copy token 被 grammar 禁止、首/中途拒绝都须通过。若尚未正确接入混合 q，随机服务路径保持 `blocked`；不能把用户采样改成 greedy。原基线本来 greedy 时，可先验证同一 greedy 策略的等价输出。

## 七步实施合同

1. **锁定当前最佳基线。** 补本机 runner、K、MTP 权重 revision、10240 宽 HC residual 与 sampling hidden 的实际合同、所有 dtype/math、Graph 和采样设置。若已含同等混合 proposer，记 `baseline-covered`。切 runner 或升级引擎另建基线，不纳入本卡收益。
2. **先做 shadow lookup。** 从已有已提交 completion 记录测 8-token、4-token 精确 suffix 的命中率、续写长度、查表/传输耗时；不把 prompt/prefix 恢复、被拒草稿或取消后未提交输出塞入 corpus。使用现有可比历史预算和 request 边界；共享 corpus 的范围沿用当前授权会话/部署策略。命中不足以抵消查找开销则停止。
3. **固定一个简单规则。** 首版只“优先 8-token，否则 4-token，取最近匹配；最多替换 K 位”，不加自适应长度、质量打分或成本 router。交付规则与 corpus 复位协议。lookup 产物携带 request generation、期望 consumed 位置；slot 复用或位置不符时全部回退原 MTP。
4. **逐步替换并记录 q。** 保留 MTP forward；每步得到候选后应用 copy mask，并用实际替换结果递推下一步 hidden/token/cache。给 copy 位接点质量表示，MTP 位接原 q，送同一 verifier。新增 metadata 是项目私有设计，不是上游现成 flag。不得把复制前的 hidden/cache 当作复制后的历史。
5. **tiny correctness 与状态检查。** CPU 小词表枚举/随机分布检验先过，再测无命中/全命中/中途拒绝、EOS、两请求交错、取消和 slot 复用。提交目标接受前缀后修复 draft cache；保持 target GDN、QSA 压缩尾部、PLE conv 的既有 rollback。验证未接受 token 不进入 corpus，copy generation 不跨请求泄漏。
6. **单变量 A/B。** A 原生 MTP；B 同 K、同图/缓冲合同下打开 copy 选择，原 forward 仍跑。若两边新增框架成本不同，先统一空 copy 分支建立 A。冷热 corpus 分开、相同 replay 初始化；至少三次交错真实请求，包含重复代码/模板主场景与低重复回归，不筛掉无命中请求。
7. **交付与回滚。** 提交 q 的数值证据、state 回归、固定规则、实现 diff、各次 CSV 与失败计数。无净收益或尾延迟变差即退出。关闭私有 copy dispatch 后回到原 MTP；释放 copy metadata/corpus 新增部分，不改变已有 KV、prefix cache 或用户采样。

## 判定与观测

记录完整轮耗时 C 和实际发出 token 数 E。净加速相对已有最佳 MTP：`S = (E_mix/C_mix) / (E_best/C_best)`。首版新增查找与 q 处理成本，未减少 MTP 网络计算，故必须由实际接受长度改善覆盖成本；相对无 MTP 的数字只作诊断。

报告 history 命中/精确匹配率、copy offered/accepted、MTP accepted、各位置拒绝率、lookup/H2D/q 处理耗时、完整 round、CPU/GPU memory、TTFT/TPOT/stream gap、输出吞吐、错误率和质量。随机文本允许因 RNG 消耗变化而不同，但分布与任务合同不能改变。当前结果均 `unknown`。
