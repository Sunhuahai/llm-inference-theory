# 夜间维护任务：已准备，待仓库发布后启用

本文件是任务定义，**不代表定时任务已创建**。预期时区 Asia/Shanghai，晚间弹性执行，参考时间 19:00（用户未指定精确时刻）。创建工具回执成功后记录真实任务 ID 与时段。

## 任务标题

更新 H800 推理优化笔记

## 每次运行提示词

```text
维护用户的公开 GitHub 仓库 Sunhuahai/qwen38-next-h800-tricks。
先通过已连接 GitHub 读取最新 main 的 AGENTS.md、README.md、environment.yaml、
docs/model-and-support.md、docs/benchmark-protocol.md、docs/feedback-loop.md，
以及本周/上周的研究、任务、决策、feedback/ 和 results/。

目标是用户实际部署的千问3.8 Next 8-bit 在H800上的推理性能；初始匹配模型为
Qwen/Qwen3.8-Flash-Next-FP8，最终以用户确认的完整ID/revision为准。
固定现有checkpoint和所有权重、激活、KV/indexer/SSM/PLE精度。
不研究重新量化/更低比特、剪枝蒸馏，不换模型，不通过关闭thinking、缩短输出、
改变任务或增加GPU预算制造性能收益。关注TTFT、TPOT、output tokens/s、
input+output total tokens/s及质量/失败/尾延迟约束。

读取此次执行可见的新增用户反馈。若相关反馈在此前对话，只有可用个人上下文确能取到时
才引用；不假设能读任意私聊。仓库记录是事实来源。无新增反馈要明确说明，
不得补造实验数据；可以核对相关上游更新，不为凑数量新建trick。

有反馈时按协议检查可比性，保存必要脱敏摘要、证据来源、baseline/candidate ID，
保留负结果，更新状态与decisions.md。公开前去除凭据、内部地址及私密prompt，
用户明确标私密的内容不公开。未复核的口述结果标user_report，不自动标validated。

针对已识别瓶颈继续查官方文档、源代码、论文、已合并PR等一手来源，记录日期和版本。
区分模型真正支持、H800硬件适配与其他GPU的微基准，不把上游数字当本机实测。
每次最多给2-3个有依据的下一步；可为复查现有候选而非新增。
每个候选写机制、适用前提、实施步骤、单变量A/B、固定项、指标、正确性和回滚。
实施由用户指派的其他Agent完成；本任务不启动GPU作业、不修改推理生产部署。

按Asia/Shanghai计算ISO周（周一至周日，使用ISO year）。若是新周且目录不存在，
创建weeks/YYYY-Www/及README.md、research.md、agent-tasks.md、feedback/README.md、
results/README.md、decisions.md和需要的新tricks/。延续项链接原ID，不复制成新发现。
同一周后续运行更新原目录；不覆盖历史反馈。上周总结保留有效、负结果与未决项。

通过已连接GitHub提交本项目授权的文档更新；先读最新head并采用安全fast-forward，
并发冲突先重新合并，不强推覆盖其他Agent工作。若不能读取/写入仓库，明确报告阻塞，
保留已完成文档，不声称提交成功。

最后用中文给用户简短更新：收到的反馈、当前判断、下一批最多3个任务、
本周目录/提交链接；无新数据时简短邀请反馈，不编造收益。无需重复请求本项目
已授权的例行文档更新许可，不向其他人发消息。
```

第一次启用时按实际当前日期安排下一个晚间时段，不能照抄历史 DTSTART。只需一个每天维护任务；它会在跨周时创建新目录。
