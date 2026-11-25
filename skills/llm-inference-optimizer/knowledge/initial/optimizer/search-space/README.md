<!-- Generated seed snapshot; source: optimizer/search-space/README.md @ 1b4c5a37763475eb39b31dd083fcad256fc2578e. Do not hand-edit. -->
# 优化空间与候选表示

> 状态：研究数据契约。此处定义完整目标接口；[离线原型](../systems/README.md)仅实现同一基线上的有限独立候选，不实现代码生成与补丁组合。

## 1. 以作用机制组织空间

候选 $a$ 应描述从当前实现 $x$ 到 $x'=a(x)$ 的变化。技术名称仅用于检索，搜索依据应是资源变化、依赖变化和适用条件。[推理理论总纲](../../theory/foundations/unified-theory.md)提供共同的物理量。

| 机制 | 主要变化 | 典型新增代价 |
| --- | --- | --- |
| 消除冗余计算 | $\Delta F<0$ | 查表、缓存、同步 |
| 减少中间访存 | $\Delta D_r<0$ | 寄存器压力、spill、布局转换 |
| 提高映射效率 | 有效计算/带宽提高 | 形状特化、编译成本 |
| 隐藏通信/拷贝 | 暴露的关键路径缩短 | buffer、资源争用、启动/排空 |
| 改善状态驻留 | 峰值容量或驻留时间下降 | 分页、搬运、恢复 |
| 改变调度 | 空闲、干扰和等待变化 | 公平性、尾延迟、缓存局部性 |
| 提高有效提交数 | 每轮提交 token 增多 | draft、verify、回滚状态 |

同一候选可改变多个量。减少逻辑字节不等于减少 HBM 流量；减少 Kernel 时间也不等于缩短服务关键路径。Ansor 的分层程序空间是结构化候选表示的参考，本表是面向推理系统的归纳。[1]

## 2. 最小候选记录

```yaml
candidate_id: C-example
parent_id: baseline-example
base_commit: unknown
state: proposed
evidence_origin: hypothesis
target_layer: kernel
diagnosis_refs: [diag-example]
mechanism: remove_intermediate_memory_traffic
allowed_edits: [selected-kernel-source]
contract:
  model_revision: unknown
  dtypes: unchanged
  sampling_semantics: unchanged
  workload_id: unknown
preconditions: []
dependencies: []
conflicts: []
effect:
  flops_delta: unknown
  bytes_delta_by_boundary: unknown
  peak_memory_delta_bytes: unknown
  critical_path_change: unknown
  extra_work: unknown
prediction:
  objective: unknown
  estimate: unknown
  uncertainty_kind: scenario
  model_version: unknown
budget:
  estimated_gpu_seconds: unknown
  hard_gpu_seconds_cap: unknown
  agent_tokens_cap: unknown
validation:
  oracle_ref: unknown
  holdout_ref: unknown
  evaluator_version: unknown
```

字段用于约束实验，不要求一开始填满。关键字段未知时应明确降级为“待测机制假设”；不能用零替代未知。

## 3. 空间中的三类约束

**确定约束**来自任务契约与逻辑条件，例如维度不可整除、所需后端不存在、固定精度被改变。这些可以在实现前拒绝。

**资源约束**来自容量及执行模型。严格已证的下界超过硬预算可以拒绝；带误差的内存/性能估计只能筛选或安排验证，不能无条件删除候选。

**条件约束**来自其他候选，例如 B 依赖 A 的布局，而 C 与 A 修改同一个后端。依赖存在不表示组合有效；仍需在组合后的完整实现上重新评价。

候选组合一般构成有向图。父指针描述代码来源，依赖边描述可执行条件，冲突边描述已知不能同时启用；三种关系应分别保存。更换父版本后，候选身份或版本也应更新，因为边际收益是相对于父版本定义的。

## 4. 搜索状态与证据状态分开

| 工作流状态 | 进入条件 | 对最佳候选的影响 |
| --- | --- | --- |
| proposed | 机制与候选已登记 | 无 |
| running | 已锁定预算、基线和评价配置 | 无 |
| failed | 编译、运行或正确性失败 | 无，记录原因 |
| evaluated | 完成约定层级评价 | 仅得到候选证据 |
| confirmed | 独立确认及约束通过 | 可进入有效候选集合 |
| rejected / inconclusive | 回归或证据不足 | 保留历史，不更新最优 |

`state=confirmed` 仍必须附上 `evidence_origin`。合成回放可确认控制规则，却不能生成 `measured` 的 GPU 性能结论；真实候选经独立复测后才可按仓库规则标为 `validated`。

## 5. 搜索空间如何扩展

先比较已有配置与后端，再开放结构化程序变换，最后开放跨模块代码修改。每扩展一级，显式报告空间变化；若实验组可以改代码而对照组只能枚举两个参数，无法把全部优势归因于理论。

高维连续参数、离散后端与开放代码没有统一的可枚举集合。初期用有限问题族做可重复评价，开放空间另外报告搜索到的程序、多样性和失败率，不声称穷尽最优。

## 参考资料

1. Zheng et al. [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng), OSDI 2020。来源核验 2026-11-25。
