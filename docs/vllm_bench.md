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
 --no-enable-log-requests \ #禁用请求级别日志
 --no-enable-prefix-caching \ #禁用前缀缓存
```

## vllm bench
```

```
