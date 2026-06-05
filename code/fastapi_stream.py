from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, List
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import json

#----------1.配置参数------------
MODEL_NAME = "/data/Qwen3.5-4B"  # 加载的模型名称
engine = None  # 延迟初始化

#----------2.定义请求和响应的数据模型-----------
class GenerateRequest(BaseModel):
    prompt: str  # 输入文本
    max_tokens: Optional[int] = 128  # 最大生成token数，默认128
    temperature: Optional[float] = 0.7  # 温度控制随机性，默认0.7
    top_p: Optional[float] = 0.9  # Top-p采样，默认0.9
    stop: Optional[List[str]] = None  # 停止词列表，默认无

class GenerateResponse(BaseModel):
    text: str  # 生成文本
    prompt: str  # 原始提示词
    finish_reason: Optional[str] = None  # 停止原因（"stop"、"length"等）

class StreamGenerateRequest(BaseModel):
    prompt: str  # 输入文本
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None
    stream: Optional[bool] = True  # 流式标志

#----------3.FastAPI生命周期管理----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用启动时初始化LLM引擎，关闭时清理资源
    """
    global engine
    print("正在初始化AsyncLLMEngine...")
    
    # 创建异步引擎
    try:
        # 使用 AsyncEngineArgs
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            trust_remote_code=True,
            max_model_len=1024,
            gpu_memory_utilization=0.5,
            # max_num_seqs=16,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("AsyncLLMEngine初始化完成")
    except Exception as e:
        print(f"引擎初始化失败: {e}")
        raise
    
    yield
    
    print("正在关闭AsyncLLMEngine...")
    # vLLM引擎会自动清理资源

#----------4.创建FastAPI应用----------
app = FastAPI(
    title="vLLM Async Inference API",
    description="异步推理服务（支持流式）",
    lifespan=lifespan
)

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """非流式生成接口"""
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎尚未初始化，请稍后重试")

    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop
    )

    try:
        # AsyncLLMEngine.generate() 返回 async generator，需要迭代
        request_id = "non-stream"
        result = None
        
        async for output in engine.generate(
            prompt=req.prompt,
            sampling_params=sampling_params,
            request_id=request_id
        ):
            result = output  # 保存最后一次迭代的结果（finished=True）
            if output.finished:
                break
        
        if result is None:
            raise HTTPException(status_code=500, detail="生成超时或无结果")
        
        text = result.outputs[0].text
        finish_reason = result.outputs[0].finish_reason

        return GenerateResponse(
            text=text,
            prompt=req.prompt,
            finish_reason=finish_reason
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.post("/generate/stream")
async def generate_stream_endpoint(req: StreamGenerateRequest):
    """流式生成接口（SSE格式）"""
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎尚未初始化，请稍后重试")

    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop
    )

    async def generate_stream():
        """SSE流式生成器（逐token产出）"""
        request_id = f"stream-{id(req)}"
        
        try:
            # AsyncLLMEngine流式生成：使用async for迭代输出
            async for output in engine.generate(
                prompt=req.prompt,
                sampling_params=sampling_params,
                request_id=request_id
            ):
                # output是RequestOutput，包含当前迭代的所有token
                # vLLM的流式输出：每次迭代返回一个或多个新生成的token
                # output.outputs[0].text表达的是累积文本
                
                if output.finished:
                    # 生成结束，发送最终标记
                    yield f"data: {json.dumps({'finish_reason': output.outputs[0].finish_reason, 'text': output.outputs[0].text}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    # 发送累积文本
                    yield f"data: {json.dumps({'text': output.outputs[0].text, 'finished': False}, ensure_ascii=False)}\n\n"
                    
        except Exception as e:
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "ok"}

#----------5.启动服务-----------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
