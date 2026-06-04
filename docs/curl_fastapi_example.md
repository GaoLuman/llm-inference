## 调用py脚本
```
CUDA_VISIBLE_DEVICES=3 python3 fastapi_vllm.py #选择3号卡
INFO:     Started server process [4569]
INFO:     Waiting for application startup.
正在初始化LLM模型...
INFO 06-04 12:27:32 [utils.py:278] non-default args: {'trust_remote_code': True, 'max_model_len': 1024, 'gpu_memory_utilization': 0.5, 'disable_log_stats': True, 'model': '/data/Qwen3.5-4B'}
WARNING 06-04 12:27:32 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_BUILD_URL
WARNING 06-04 12:27:32 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_IMAGE_TAG
WARNING 06-04 12:27:32 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_BUILD_PIPELINE
WARNING 06-04 12:27:32 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_BUILD_COMMIT
INFO 06-04 12:27:32 [model.py:617] Resolved architecture: Qwen3_5ForConditionalGeneration
INFO 06-04 12:27:32 [model.py:1752] Using max model len 1024
INFO 06-04 12:27:32 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 06-04 12:27:32 [vllm.py:977] Asynchronous scheduling is enabled.
INFO 06-04 12:27:32 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
[transformers] `Qwen2VLImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `Qwen2VLImageProcessor` instead.
[transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
WARNING 06-04 12:27:45 [system_utils.py:157] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing for more information. Reasons: CUDA is initialized
(EngineCore pid=4775) INFO 06-04 12:27:55 [core.py:112] Initializing a V1 LLM engine (v0.22.0) with config: model='/data/Qwen3.5-4B', speculative_config=None, tokenizer='/data/Qwen3.5-4B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=1024, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=/data/Qwen3.5-4B, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'ir_enable_torch_wrap': True, 'splitting_ops': ['vllm::unified_attention_with_output', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::qwen_gdn_attention_core', 'vllm::gdn_attention_core_xpu', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::deepseek_v4_attention', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']), enable_flashinfer_autotune=True, moe_backend='auto', linear_backend='auto')
(EngineCore pid=4775) [transformers] `Qwen2VLImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `Qwen2VLImageProcessor` instead.
(EngineCore pid=4775) INFO 06-04 12:27:59 [parallel_state.py:1422] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.19.0.8:36851 backend=nccl
(EngineCore pid=4775) INFO 06-04 12:27:59 [parallel_state.py:1735] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=4775) INFO 06-04 12:28:00 [topk_topp_sampler.py:45] Using FlashInfer for top-p & top-k sampling.
(EngineCore pid=4775) [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
(EngineCore pid=4775) INFO 06-04 12:28:05 [gpu_model_runner.py:5037] Starting to load model /data/Qwen3.5-4B...
(EngineCore pid=4775) INFO 06-04 12:28:06 [cuda.py:433] Using backend AttentionBackendEnum.FLASH_ATTN for vit attention
(EngineCore pid=4775) INFO 06-04 12:28:06 [mm_encoder_attention.py:372] Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.
(EngineCore pid=4775) INFO 06-04 12:28:06 [qwen_gdn_linear_attn.py:228] Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=None).
(EngineCore pid=4775) INFO 06-04 12:28:06 [cuda.py:378] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=4775) INFO 06-04 12:28:06 [flash_attn.py:636] Using FlashAttention version 2
(EngineCore pid=4775) INFO 06-04 12:28:08 [weight_utils.py:922] Filesystem type for checkpoints: EXT4. Checkpoint size: 8.68 GiB. Available RAM: 870.86 GiB.
(EngineCore pid=4775) INFO 06-04 12:28:08 [weight_utils.py:945] Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  1.21it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.30it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.29it/s]
(EngineCore pid=4775) 
(EngineCore pid=4775) INFO 06-04 12:28:10 [default_loader.py:397] Loading weights took 1.64 seconds
(EngineCore pid=4775) INFO 06-04 12:28:10 [gpu_model_runner.py:5132] Model loading took 8.61 GiB memory and 4.171419 seconds
(EngineCore pid=4775) INFO 06-04 12:28:10 [interface.py:649] Setting attention block size to 528 tokens to ensure that attention page size is >= mamba page size.
(EngineCore pid=4775) INFO 06-04 12:28:10 [interface.py:673] Padding mamba page size by 0.76% to ensure that mamba page size and attention page size are exactly equal.
(EngineCore pid=4775) INFO 06-04 12:28:11 [gpu_model_runner.py:6136] Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum feature size.
(EngineCore pid=4775) INFO 06-04 12:28:14 [backends.py:1089] Using cache directory: /root/.cache/vllm/torch_compile_cache/705999a272/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=4775) INFO 06-04 12:28:14 [backends.py:1148] Dynamo bytecode transform time: 1.47 s
(EngineCore pid=4775) INFO 06-04 12:28:16 [backends.py:292] Directly load the compiled graph(s) for compile range (1, 8192) from the cache, took 1.874 s
(EngineCore pid=4775) INFO 06-04 12:28:16 [decorators.py:311] Directly load AOT compilation from path /root/.cache/vllm/torch_compile_cache/torch_aot_compile/cb152bee93d53efe67aa9c96affc90d5334d771d9ef4c0660a0b5f8cac5d2810/rank_0_0/model
(EngineCore pid=4775) INFO 06-04 12:28:16 [monitor.py:53] torch.compile took 3.59 s in total
(EngineCore pid=4775) INFO 06-04 12:28:18 [monitor.py:81] Initial profiling/warmup run took 2.33 s
(EngineCore pid=4775) INFO 06-04 12:28:19 [gpu_model_runner.py:6279] Profiling CUDA graph memory: PIECEWISE=51 (largest=512), FULL=35 (largest=256)
(EngineCore pid=4775) INFO 06-04 12:28:22 [gpu_model_runner.py:6365] Estimated CUDA graph memory: 0.64 GiB total
(EngineCore pid=4775) INFO 06-04 12:28:23 [gpu_worker.py:466] Available KV cache memory: 11.11 GiB
(EngineCore pid=4775) INFO 06-04 12:28:23 [gpu_worker.py:481] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.5000 is equivalent to --gpu-memory-utilization=0.4855 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.5145. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=4775) INFO 06-04 12:28:23 [kv_cache_utils.py:1733] GPU KV cache size: 141,107 tokens
(EngineCore pid=4775) INFO 06-04 12:28:23 [kv_cache_utils.py:1734] Maximum concurrency for 1,024 tokens per request: 137.80x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:10<00:00,  4.86it/s]
Capturing CUDA graphs (decode, FULL): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:05<00:00,  5.97it/s]
(EngineCore pid=4775) INFO 06-04 12:28:40 [gpu_model_runner.py:6456] Graph capturing finished in 17 secs, took 0.55 GiB
(EngineCore pid=4775) INFO 06-04 12:28:40 [gpu_worker.py:619] CUDA graph pool memory: 0.55 GiB (actual), 0.64 GiB (estimated), difference: 0.1 GiB (17.9%).
(EngineCore pid=4775) INFO 06-04 12:28:41 [jit_monitor.py:54] Kernel JIT monitor activated — Triton JIT compilations during inference will be logged as warnings.
(EngineCore pid=4775) INFO 06-04 12:28:41 [core.py:302] init engine (profile, create kv cache, warmup model) took 30.23 s (compilation: 3.59 s)
(EngineCore pid=4775) INFO 06-04 12:28:41 [vllm.py:977] Asynchronous scheduling is enabled.
(EngineCore pid=4775) INFO 06-04 12:28:41 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
LLM模型初始化完成
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.90it/s]
Processed prompts:   0%|                                                                                                                | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s](EngineCore pid=4775) WARNING 06-04 12:29:15 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _zero_kv_blocks_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
(EngineCore pid=4775) WARNING 06-04 12:29:15 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
(EngineCore pid=4775) WARNING 06-04 12:29:15 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _causal_conv1d_fwd_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
(EngineCore pid=4775) WARNING 06-04 12:29:15 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _fused_post_conv_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.59s/it, est. speed input: 2.52 toks/s, output: 63.02 toks/s]
INFO:     127.0.0.1:51170 - "POST /generate HTTP/1.1" 200 OK
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 722.04it/s]
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.30s/it, est. speed input: 0.60 toks/s, output: 79.89 toks/s]
INFO:     127.0.0.1:51182 - "POST /generate HTTP/1.1" 200 OK
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 761.35it/s]
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.27s/it, est. speed input: 3.14 toks/s, output: 78.51 toks/s]
INFO:     127.0.0.1:40888 - "POST /generate HTTP/1.1" 200 OK
Rendering prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 710.06it/s]
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:10<00:00, 10.57s/it, est. speed input: 0.47 toks/s, output: 79.70 toks/s]
INFO:     127.0.0.1:54828 - "POST /generate HTTP/1.1" 200 OK
^C(EngineCore pid=4775) INFO 06-04 12:30:26 [core.py:1266] Shutdown initiated (timeout=0)
(EngineCore pid=4775) INFO 06-04 12:30:26 [core.py:1289] Shutdown complete
INFO:     Shutting down
INFO:     Waiting for application shutdown.
正在关闭LLM模型...
INFO:     Application shutdown complete.
INFO:     Finished server process [4569]
/usr/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
  ```

## curl命令
```
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt":"南京有什么美食?","max_tokens":100}'
{"text":"\n\n南京（Nanjing）作为六朝古都，有着“金陵”的美誉，其美食文化非常丰富，既有深厚的历史底蕴，又有独特的淮扬风味和苏帮菜特色。\n\n以下为您整理的南京必吃美食清单，按类别划分：\n\n### 1. 南京“四大名菜” (必点经典)\n这几道菜是南京饮食文化的代表，通常出现在宴席上。\n\n*   **盐水鸭 (Yan Shui Ya)**\n    *","prompt":"南京有什么美食?","finish_reason":"length"}  

curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt":"介绍一下什么是奥林匹克运动会?","max_tokens":1000}'
{"text":"\n\n<think>\n\n</think>\n\n**奥林匹克运动会**（Olympic Games）是世界上规模最大的综合性体育赛事，被誉为“世界体育的盛会”。它起源于古希腊，是现代体育精神的最高象征，由国际奥委会（IOC）负责组织和推广。\n\n以下是对奥林匹克运动会的详细介绍：\n\n### 1. 起源与历史\n*   **古代起源**：起源于公元前 776 年的古希腊奥林匹亚，最初是为了祭祀众神宙斯而举行的宗教庆典，主要项目包括短跑、摔跤、赛马等。\n*   **现代复兴**：1894 年，在法国巴黎举行的“国际体育代表大会”上，法国教育家皮埃尔·德·顾拜旦（Pierre de Coubertin）正式倡议恢复奥运会。\n*   **首届现代奥运会**：1896 年 4 月 6 日，首届现代夏季奥林匹克运动会在希腊雅典举行，共有 14 个国家、241 名运动员参加。\n\n### 2. 基本架构\n现代奥运会主要分为两个系列，每四年交替举行（在特殊情况下会合并或调整）：\n*   **夏季奥林匹克运动会**：通常在 7 月至 8 月举行，涵盖田径、游泳、篮球、足球等约 30 个大项。\n*   **冬季奥林匹克运动会**：通常在 2 月至 3 月举行，涵盖冰雪运动，如滑雪、滑冰、冰球等。\n*   **青年奥林匹克运动会**：为 14-18 岁的青少年设立的赛事，旨在培养年轻一代的体育精神。\n\n### 3. 核心精神\n奥运会不仅仅是一场比赛，更是一种文化现象，其核心精神概括为：\n*   **卓越（Excellence）**：追求卓越，不断突破个人和团队的极限。\n*   **友谊（Friendship）**：促进各国人民之间的理解、友谊与合作。\n*   **尊重（Respect）**：尊重规则、对手以及所有参与者。\n*   **和平（Peace）**：通过体育传递和平理念，超越政治、宗教和种族隔阂。\n\n顾拜旦曾名言：\"**参与比取胜更重要**\"（Participation is more important than victory），强调了体育中奋斗过程的价值。\n\n### 4. 重要标志与元素\n*   **五环标志**：由蓝、黄、黑、绿、红五种颜色的圆环组成，象征五大洲的团结和全世界运动员的友好相聚。\n*   **圣火**：象征光明与希望，通常在希腊奥林匹亚通过火炬接力方式传至主办国。\n*   **圣火台**：在开幕式上点燃主火炬塔，象征奥运精神的永恒。\n*   **吉祥物**：每届奥运会都会设计独特的吉祥物，代表该届的主题和特色。\n\n### 5. 组织与规则\n*   **国际奥委会（IOC）**：成立于 1894 年，总部设在瑞士洛桑，是奥林匹克运动的最高决策机构，负责制定规则、选拔主办城市等。\n*   **主办城市**：每届奥运会由一个城市申办并主办，该城市需承担巨大的筹备工作，包括场馆建设、安保、医疗等。\n*   **反兴奋剂**：世界反兴奋剂机构（WADA）与国际奥委会紧密合作，严禁运动员使用违禁药物，确保比赛的公平性。\n\n### 6. 意义与影响\n奥林匹克运动会是全球性的盛会，具有深远的社会影响：\n*   **展示国家形象**：通过体育竞技展示主办国乃至国家的文化、实力和软实力。\n*   **促进经济发展**：带动主办国的旅游、交通、建筑和媒体等相关产业。\n*   **推动体育普及**：激发全球民众参与体育运动的兴趣，提升健康意识。\n*   **和平交流**：为不同背景的人们提供了一个难得的交流平台，有助于增进国际理解。\n\n总之，奥林匹克运动会不仅是体育竞技的巅峰，更是人类追求和平、友谊与卓越精神的集中体现。","prompt":"介绍一下什么是奥林匹克运动会?","finish_reason":"stop"}
```
