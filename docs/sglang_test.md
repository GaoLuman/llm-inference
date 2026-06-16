```
python -m sglang.launch_server  \
--model-path /data/Qwen3.5-2B  \
--host 0.0.0.0   \
--port 30000 \
--chunked-prefill-size 4096
```

```
python -m sglang.bench_serving  \
--backend sglang \ 
--host localhost  \ 
--port 30000  \
--dataset-name random-ids \
--random-input-len 1024  \
--random-output-len 128  \
--num-prompts 50  \
--max-concurrency 8 \
--request-rate inf \
--warmup-requests 10 \
--seed 123456  
```

```
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 8         
Successful requests:                     50        
Benchmark duration (s):                  5.03      
Total input tokens:                      27927     
Total input text tokens:                 27927     
Total generated tokens:                  3469      
Total generated tokens (retokenized):    3495      
Request throughput (req/s):              9.94      
Input token throughput (tok/s):          5552.96   
Output token throughput (tok/s):         689.77    
Peak output token throughput (tok/s):    789.00    
Peak concurrent requests:                19        
Total token throughput (tok/s):          6242.74   
Concurrency:                             7.36      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   740.79    
Median E2E Latency (ms):                 772.57    
P90 E2E Latency (ms):                    1216.16   
P99 E2E Latency (ms):                    1332.22   
---------------Time to First Token----------------
Mean TTFT (ms):                          67.04     
Median TTFT (ms):                        45.58     
P99 TTFT (ms):                           168.14    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.61      
Median TPOT (ms):                        9.91      
P99 TPOT (ms):                           12.64     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.85      
Median ITL (ms):                         7.06      
P95 ITL (ms):                            39.17     
P99 ITL (ms):                            70.38     
Max ITL (ms):                            185.64    
==================================================
```
