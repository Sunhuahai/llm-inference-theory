# Txxx · 候选名称

- 状态：`researched` / `conditional`
- 调研日期：YYYY-MM-DD
- 目标指标与优先级：
- 证据来源：literature / hypothesis；本机实测：无

## 问题与机制

解释哪部分时间或资源被节省。给出一手来源 URL；资料报告的硬件、版本和微基准范围与本机推断分开。

## 适用前提

精确模型、框架 commit、卡数/互联、dtype、业务与功能限制。检查基线是否已覆盖。

## 实施 Agent 操作

写清需要检查的模块/日志/形状、最小改动和配置差异；不能把未验证的 flags 写成现成可运行命令。

## 最小 A/B

唯一变量、固定项、主场景与回归场景、至少三次重复、TTFT/TPOT/吞吐与解释性 profile。

## 验收与回滚

质量、正确性、尾延迟、错误率、显存、收益超过噪声。未给业务 SLO 时报告 Pareto 权衡，不自造硬阈值。

## 结果与下一步

未知就填 unknown。引用 baseline/candidate IDs、证据文件、实施 commit；指出仍不能解释的现象。
