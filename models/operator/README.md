# 算子性能模型：从资源成本到形状相关的时间

> 状态：`derived` / 模型设计，2026-11-25。尚无本仓库新采集的 GPU 校准数据。

## 1. 为什么不能使用一个全局效率系数

同样的 FLOPs 会被映射成不同数量的 tile、warp 与内存访问。小矩阵可能没有足够并行度，大矩阵可能接近稳定吞吐，融合还会改变寄存器和共享内存压力。NVIDIA 的 [矩阵乘法指南][gemm]明确讨论算术强度、tile 与形状效率；本文不将其中任何硬件数字移植到用户设备。

令特征为 $x=(M,N,K,\text{dtype},\text{layout},\text{backend},\text{device})$。一个可解释的近似为：

$$
\widehat\tau(x)=a(x)+\max\left(\frac{F(x)}{P_{\rm eff}(x)},\frac{D(x)}{B_{\rm eff}(x)}\right)+r(x).
$$

$a$ 为明确不重叠的固定成本，$r$ 为经过验证的残差项。该式是建模选择，不是时间恒等式。若同一次 profiler 的 launch gap 已进入执行图，就不能再加进 $a$。

## 2. 四类算子的输入重点

| 算子 | 资源模型 | 校准应覆盖什么 |
| --- | --- | --- |
| GEMM | 常规稠密乘法 $F\approx2MNK$；读写按布局和缓存边界计 | 小/大 $M$、维度整除、dtype、后端与 tile |
| Attention | 合法注意力位置对、Q/K/V 搬运、中间张量 | prefill/decode、长度、head 数、mask、分页与 GQA |
| MoE | 每专家 shape、路由并集、dispatch/combine | 专家负载尾部、批内共享、top-k、每 rank 不均衡 |
| Collective | 消息大小、通信算法、进程组、拓扑 | 小消息延迟、大消息带宽、并发与暴露时间 |

合成 GEMM 例：$M=16,N=K=4096$，$F=536{,}870{,}912$ FLOPs。若输入、权重和输出均为 2 bytes 且各读写一次，理想字节数为 $33{,}816{,}576$，算术强度约 15.88 FLOP/byte。实际 HBM 流量还可能受 cache、分块重读和附加算子影响；这是账本起点，不能直接认定运行时间。

## 3. 可辨识性先于复杂拟合

如果全部样本都由带宽主导，则许多不同 $P_{\rm eff}$ 都能拟合相同时间；从这些数据无法识别计算效率。只用 $F/T$ 和 $D/T$ 同时反推两个速率，再预测同一 $T$，只会循环重现数据。

建议按机制选少量锚点：一个带宽敏感形状、一个计算敏感形状、一个小形状固定开销点，再保留未参与拟合的 shape。对强非线性后端切换，应分段或使用查表模型，不能在 tile 边界上盲目平滑外推。有效速率的适用域至少包含硬件、精度、后端、形状和并发条件。

## 4. 从局部预测走向全局

算子输出需包括预测时长、情景区间或经校准区间、是否超出适用域，以及新增容量需求。单算子从 8 ms 降到 4 ms，只说明局部变化；若它与 6 ms 通信重叠，对该阶段的理想收益最多由 8 ms 降为 6 ms。全局结果必须经过[执行图](../execution/README.md)重算。

可证伪目标：在留出 shape 上，预测应解释优化前后的变化方向；只拟合基线总时间不够。如果错误集中在某类 shape，先检查 layout、kernel dispatch 和缓存条件，再增加模型复杂度。

[校准与不确定性](../calibration/README.md) · [微基准](../../experiments/microbenchmarks/README.md) · [资源账本](../resource/README.md)

[gemm]: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html "NVIDIA Matrix Multiplication Background User's Guide; 核对于 2026-11-25"
