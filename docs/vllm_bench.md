## Benchmark Results
**测试环境**
- 模型：Qwen3.5-2B
- 硬件：NVIDIA

## vllm服务端
```
vllm serve /data/Qwen3.5-2B \
 --gpu_memory_utilization 0.8 \ #GPU显存上限
 --port 8030 \
 --max-model-len 4096 \ #单次请求总token上限
 --max-num-seqs 32 \ #最大请求并发数
 --no-enable-log-requests \ #禁用请求级别日志
 --no-enable-prefix-caching \ #禁用前缀缓存
```

## vllm bench
```
vllm bench serve \
    --model /data/Qwen3.5-2B \
    --dataset-name random \ #数据集类型
    --port 8030 \
    --num-prompt 50 \ #请求总数
    --max-concurrency 8\ #服务器最大请求数
    --request-rate inf \ #请求速率，inf指无限并发
    --save-result \
    --result-dir ./benchmark_results

```
### 性能结果
|最大并发请求数|单次请求总token上限|服务器最大请求数|请求总数|输出吞吐量(tok/s)|mean_e2el_ms|mean_ttft_ms|mean_tpot_ms|
|------------|------------|------------|--------|----------|----------|--------|--------|
| 32 | 4096 | 1 | 50 | 175.08 | 730.44 | 54.56 | 5.32 |
| 32 | 4096 | 4 | 50 | 561.04 | 878.61 | 102.19 | 6.11 |
| 32 | 4096 | 16 | 50 | 1390.32 | 1238.72 | 261.50 | 7.69 |
| 32 | 4096 | 32 | 50 | 2059.95 | 1667.98 | 418.90 | 9.83 |
| 32 | 8192 | 1 | 50 | 154.81 | 826.20 | 54.81 | 6.07 |
| 32 | 8192 | 4 | 50 | 505.605 | 975.08 | 103.36 | 6.86 |
| 32 | 8192 | 16 | 50 | 1269.57 | 1354.42 | 279.07 | 8.47 |
| 32 | 8192 | 32 | 50 | 1834.04 | 1816.11 | 516.37 | 10.23 |
| 64 | 8192 | 1 | 50 | 163.35 | 783.00 | 102.47 | 5.36 |
| 64 | 8192 | 4 | 50 | 493.72 | 1003.24 | 105.18 | 7.71 |
| 64 | 8192 | 16 | 50 | 1215.27 | 1243.57 | 298.87 | 7.84 |
| 64 | 8192 | 32 | 50 | 2113.48 | 1577.68 | 448.84 | 8.89 |

### 量化实验
```
vllm serve /data/Qwen3.5-2B \
 --quantization bitsandbytes \
 --gpu_memory_utilization 0.8 \
 --port 8030 \
 --max-model-len 4096 \
 --no-enable-log-requests \
 --no-enable-prefix-caching
```
第二种量化方法，采用已经量化的模型
```
vllm serve /data/Qwen3.5-2B-W8A8-Dynamic-Per-Token \
--gpu_memory_utilization 0.8 \
--port 8030 \
--max-model-len 4096 \
--no-enable-log-requests \
--no-enable-prefix-caching
```

|模型|精度|显存|吞吐量(tok/s)|Mean TTFT(ms)|P99 TTFT(ms)|Mean TPOT(ms)|P99 TPOT(ms)|Mean ITL(ms)|P99 ITL(ms)|
|-----|-----|-----|-----|-----|-----|------|------|-----|-----|
|Qwen3.5-2B|fp16|4.25GB|860.40|141.58|207.10|7.40|7.86|7.40|11.37|
|Qwen3.5-2B|int8|1.98G|825.6|215.39|318.57|7.23|8.0|7.23|11.39|
|Qwen3.5-2B-W8A8-Dynamic-Per-Token|INT8|2.97G|990.52|170.8|264.26|6.05|6.66|6.05|8.73|

- bitsandbytes是一种实时量化方法，需要模型加载时启动量化，启动速度会变慢。  
- 本案例中发现量化后的ttft升高而tpot降低。ttft由Prefill阶段决定，tpot由Decode阶段决定，量化后的模型Prefill阶段需要做反量化计算，反而可能增高。但是由于带宽压力减小，Decode阶段会变快。  
- 量化不一定会增加吞吐量，因为降低了带宽，大部分时候会增加。但是如果吞吐量的限制不在带宽而是在Kernel启动延时等开销上时，由于反量化，可能还会降低。要结合实际情况来看

```
vllm serve /data/Qwen3.5-2B \
 --gpu_memory_utilization 0.95 \
 --port 8030 \
 --max-model-len 4096 \
 --no-enable-log-requests \
 --no-enable-prefix-caching
```
```
vllm serve /workspace/code/models/Qwen3.5-2B-AWQ-4bit \
 --gpu_memory_utilization 0.95 \
 --port 8030 \
 --max-model-len 4096 \
 --no-enable-log-requests \
 --no-enable-prefix-caching
```
|模型|精度|显存|吞吐量(tok/s)|Mean TTFT(ms)|P99 TTFT(ms)|Mean TPOT(ms)|P99 TPOT(ms)|Mean ITL(ms)|P99 ITL(ms)|
|-----|-----|-----|-----|-----|-----|------|------|-----|-----|
|Qwen3.5-2B|fp16|4.25GB|949.51|136.95|197.04|6.64|7.07|6.64|8.17|
|Qwen3.5-2B-AWQ-4bit|int4|2.43GB|1421.6|139.86|199.79|4.08|4.64|4.08|5.51|
