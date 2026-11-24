# 优化收益情景预测（合成演示数据，非 H800 实测）

基线：SYNTHETIC-DEMO-NOT-H800-MEASURED；TTFT/TPOT 预测为低负载均值，不是 p95。

| 情景 | TTFT ms | TTFT 加速 | TPOT ms | TPOT 加速 | 饱和输出 tok/s | 容量加速 | 新瓶颈 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 150.00 | 1.00× | 22.00 | 1.00× | 1800.00 | 1.00× | 合成基线 |
| T003-QSA-conservative | 148.33 | 1.011× | 20.25 | 1.086× | 1894.74 | 1.053× | gpu_compute |
| T003-QSA-optimistic | 145.94 | 1.028× | 18.60 | 1.183× | 2195.12 | 1.220× | gpu_compute |
| T009-MTP-hypothetical | 150.00 | 1.000× | 15.87 | 1.386× | 1578.95 | 0.877× | gpu_compute |
| T006-PLE-overlap-only | 146.97 | 1.021× | 22.00 | 1.000× | 1800.00 | 1.000× | gpu_compute |

## 适用范围

- 输出吞吐模型要求固定 workload、固定 GPU 数量，以及饱和基线与资源繁忙比例。
- 到达率不足时，实际输出 tok/s 不会因为容量增加而自动增加。
- 分块调度、多请求排队、p95/p99 和 P/D 分离需补充 trace 级仿真。
- 输入参数为情景假设；合成示例不构成 H800 加速结论。

## 预警

- T009-MTP-hypothetical：MTP 对首 token 和批处理的影响未计入 TTFT
