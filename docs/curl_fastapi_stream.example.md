## python脚本执行
```
CUDA_VISIBLE_DEVICES=0 python3 fastapi_stream.py 
INFO:     Started server process [1291]
INFO:     Waiting for application startup.
正在初始化AsyncLLMEngine...
WARNING 06-05 07:28:09 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_BUILD_URL
WARNING 06-05 07:28:09 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_IMAGE_TAG
WARNING 06-05 07:28:09 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_BUILD_PIPELINE
WARNING 06-05 07:28:09 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_BUILD_COMMIT
INFO 06-05 07:28:09 [model.py:617] Resolved architecture: Qwen3_5ForConditionalGeneration
INFO 06-05 07:28:09 [model.py:1752] Using max model len 1024
INFO 06-05 07:28:09 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=2048.
INFO 06-05 07:28:09 [vllm.py:977] Asynchronous scheduling is enabled.
INFO 06-05 07:28:09 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
[transformers] `Qwen2VLImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `Qwen2VLImageProcessor` instead.
[transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
WARNING 06-05 07:28:22 [system_utils.py:157] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing for more information. Reasons: CUDA is initialized
(EngineCore pid=1497) INFO 06-05 07:28:29 [core.py:112] Initializing a V1 LLM engine (v0.22.0) with config: model='/data/Qwen3.5-4B', speculative_config=None, tokenizer='/data/Qwen3.5-4B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=1024, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=/data/Qwen3.5-4B, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'ir_enable_torch_wrap': True, 'splitting_ops': ['vllm::unified_attention_with_output', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::qwen_gdn_attention_core', 'vllm::gdn_attention_core_xpu', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::deepseek_v4_attention', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 256, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']), enable_flashinfer_autotune=True, moe_backend='auto', linear_backend='auto')
(EngineCore pid=1497) [transformers] `Qwen2VLImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `Qwen2VLImageProcessor` instead.
(EngineCore pid=1497) INFO 06-05 07:28:34 [parallel_state.py:1422] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.19.0.10:50715 backend=nccl
(EngineCore pid=1497) INFO 06-05 07:28:34 [parallel_state.py:1735] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=1497) INFO 06-05 07:28:35 [topk_topp_sampler.py:45] Using FlashInfer for top-p & top-k sampling.
(EngineCore pid=1497) [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
(EngineCore pid=1497) INFO 06-05 07:28:40 [gpu_model_runner.py:5037] Starting to load model /data/Qwen3.5-4B...
(EngineCore pid=1497) INFO 06-05 07:28:41 [cuda.py:433] Using backend AttentionBackendEnum.FLASH_ATTN for vit attention
(EngineCore pid=1497) INFO 06-05 07:28:41 [mm_encoder_attention.py:372] Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.
(EngineCore pid=1497) INFO 06-05 07:28:41 [qwen_gdn_linear_attn.py:228] Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=None).
(EngineCore pid=1497) INFO 06-05 07:28:41 [cuda.py:378] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=1497) INFO 06-05 07:28:41 [flash_attn.py:636] Using FlashAttention version 2
(EngineCore pid=1497) INFO 06-05 07:28:42 [weight_utils.py:922] Filesystem type for checkpoints: EXT4. Checkpoint size: 8.68 GiB. Available RAM: 843.15 GiB.
(EngineCore pid=1497) INFO 06-05 07:28:42 [weight_utils.py:945] Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  1.26it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.29it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.28it/s]
(EngineCore pid=1497) 
(EngineCore pid=1497) INFO 06-05 07:28:44 [default_loader.py:397] Loading weights took 1.63 seconds
(EngineCore pid=1497) INFO 06-05 07:28:45 [gpu_model_runner.py:5132] Model loading took 8.61 GiB memory and 3.455410 seconds
(EngineCore pid=1497) INFO 06-05 07:28:45 [interface.py:649] Setting attention block size to 528 tokens to ensure that attention page size is >= mamba page size.
(EngineCore pid=1497) INFO 06-05 07:28:45 [interface.py:673] Padding mamba page size by 0.76% to ensure that mamba page size and attention page size are exactly equal.
(EngineCore pid=1497) INFO 06-05 07:28:45 [gpu_model_runner.py:6136] Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum feature size.
(EngineCore pid=1497) INFO 06-05 07:28:48 [backends.py:1089] Using cache directory: /root/.cache/vllm/torch_compile_cache/17785336a2/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=1497) INFO 06-05 07:28:48 [backends.py:1148] Dynamo bytecode transform time: 1.54 s
(EngineCore pid=1497) INFO 06-05 07:28:50 [backends.py:292] Directly load the compiled graph(s) for compile range (1, 2048) from the cache, took 1.723 s
(EngineCore pid=1497) INFO 06-05 07:28:50 [decorators.py:311] Directly load AOT compilation from path /root/.cache/vllm/torch_compile_cache/torch_aot_compile/c05d651d19b09b1dfe1dabaa6b6acf5596ae5dc042dceaa1b14ff71a6c4f5822/rank_0_0/model
(EngineCore pid=1497) INFO 06-05 07:28:50 [monitor.py:53] torch.compile took 3.51 s in total
(EngineCore pid=1497) INFO 06-05 07:28:51 [monitor.py:81] Initial profiling/warmup run took 1.10 s
(EngineCore pid=1497) INFO 06-05 07:28:52 [gpu_model_runner.py:6279] Profiling CUDA graph memory: PIECEWISE=35 (largest=256), FULL=19 (largest=128)
(EngineCore pid=1497) INFO 06-05 07:28:57 [gpu_model_runner.py:6365] Estimated CUDA graph memory: 0.40 GiB total
(EngineCore pid=1497) INFO 06-05 07:28:57 [gpu_worker.py:466] Available KV cache memory: 12.85 GiB
(EngineCore pid=1497) INFO 06-05 07:28:57 [gpu_worker.py:481] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.5000 is equivalent to --gpu-memory-utilization=0.4915 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.5085. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=1497) INFO 06-05 07:28:57 [kv_cache_utils.py:1733] GPU KV cache size: 163,225 tokens
(EngineCore pid=1497) INFO 06-05 07:28:57 [kv_cache_utils.py:1734] Maximum concurrency for 1,024 tokens per request: 159.40x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|█| 
Capturing CUDA graphs (decode, FULL): 100%|█| 19/19 [00:05<00:00,
(EngineCore pid=1497) INFO 06-05 07:29:11 [gpu_model_runner.py:6456] Graph capturing finished in 14 secs, took 0.35 GiB
(EngineCore pid=1497) INFO 06-05 07:29:11 [gpu_worker.py:619] CUDA graph pool memory: 0.35 GiB (actual), 0.4 GiB (estimated), difference: 0.06 GiB (16.4%).
(EngineCore pid=1497) INFO 06-05 07:29:11 [jit_monitor.py:54] Kernel JIT monitor activated — Triton JIT compilations during inference will be logged as warnings.
(EngineCore pid=1497) INFO 06-05 07:29:11 [core.py:302] init engine (profile, create kv cache, warmup model) took 26.68 s (compilation: 3.51 s)
(EngineCore pid=1497) INFO 06-05 07:29:12 [vllm.py:977] Asynchronous scheduling is enabled.
(EngineCore pid=1497) INFO 06-05 07:29:12 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
AsyncLLMEngine初始化完成
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
WARNING 06-05 07:29:48 [input_processor.py:274] Passing raw prompts to InputProcessor is deprecated and will be removed in v0.18. You should instead pass the outputs of Renderer.render_cmpl() or Renderer.render_chat().
(EngineCore pid=1497) WARNING 06-05 07:29:48 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _zero_kv_blocks_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
(EngineCore pid=1497) WARNING 06-05 07:29:48 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
(EngineCore pid=1497) WARNING 06-05 07:29:49 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _causal_conv1d_fwd_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
(EngineCore pid=1497) WARNING 06-05 07:29:49 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _fused_post_conv_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
INFO:     127.0.0.1:35226 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:42822 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:56876 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:57542 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:43200 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:43418 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:42932 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:60108 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:51220 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:39078 - "POST /generate/stream HTTP/1.1" 200 OK
^C(EngineCore pid=1497) INFO 06-05 07:34:47 [core.py:1266] Shutdown initiated (timeout=0)
(EngineCore pid=1497) INFO 06-05 07:34:47 [core.py:1289] Shutdown complete
INFO:     Shutting down
INFO:     Waiting for application shutdown.
正在关闭AsyncLLMEngine...
INFO:     Application shutdown complete.
INFO:     Finished server process [2093]
/usr/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
Exception ignored in: <function AsyncLLM.__del__ at 0x7efa37dc1940>
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 257, in __del__
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 261, in shutdown
TypeError: 'NoneType' object is not callable
```
## curl执行结果
```
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt":"请简单介绍一下南京理工大学","max_tokens":1000}'
{"text":"机械工程学院的本科专业。\n\n南京理工大学机械工程学院拥有扎实的学科基础，其本科专业设置紧密围绕国家重大需求及国防科技工业发展。该学院主要开设的本科专业如下：\n\n1.  **机械电子工程**\n    这是该学院最具代表性的王牌专业之一。该专业强调“机电软”一体化，融合了机械工程、电子工程和控制科学，旨在培养具备机械设计、制造、控制及系统集成的复合型人才。其学科实力雄厚，在行业内认可度极高。\n\n2.  **机械设计制造及其自动化**\n    作为传统的机械类专业，该专业重点研究机械系统的结构、设计、制造、工艺及自动化控制技术。它是机械工程的基石，培养学生在机械制造、自动化装备等领域的应用能力。\n\n3.  **智能制造工程**\n    这是一个相对较新且极具前瞻性的专业。该专业聚焦于工业 4.0 背景下的智能制造技术，涵盖机器人技术、工业物联网、数字孪生等前沿领域，致力于培养适应未来制造业转型升级需求的高级工程技术人才。\n\n4.  **车辆工程**\n    依托南京理工大学在国防军工领域的深厚背景，该专业主要面向汽车及车辆制造行业，研究车辆的总体设计、制造、试验及控制，是汽车产业和国防交通的重要支撑专业。\n\n**总结：**\n南京理工大学机械工程学院的本科专业群以**机械电子工程**为龙头，形成了从传统机械制造到现代智能制造，从地面车辆到航空航天动力系统的完整专业布局。这些专业不仅理论扎实，而且与国防军工、航空航天及汽车制造等国家重点行业联系紧密。","prompt":"请简单介绍一下南京理工大学","finish_reason":"stop"}
```
```
curl -X POST "http://localhost:8000/generate/stream" -H "Content-Type: application/json" -d '{"prompt":"一句话介绍你自己","max_tokens":100}'
data: {"text": "\n\n", "finished": false}

data: {"text": "\n\n<think>", "finished": false}

data: {"text": "\n\n<think>\n\n", "finished": false}

data: {"text": "\n\n<think>\n\n</think>", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Q", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成及", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成及多", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成及多语言", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成及多语言理解", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成及多语言理解能力", "finished": false}

data: {"text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成及多语言理解能力。", "finished": false}

data: {"finish_reason": "stop", "text": "\n\n<think>\n\n</think>\n\n我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成及多语言理解能力。"}

data: [DONE]


```
