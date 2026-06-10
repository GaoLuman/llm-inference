from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, List
from vllm import LLM, SamplingParams

#----------1.配置参数------------
MODEL_NAME = "/data/Qwen3.5-4B" # 加载的模型名称
llm = None  # 延迟初始化

#----------2.定义请求和响应的数据模型-----------
class GenerateRequest(BaseModel):
    prompt: str                              # 输入文本
    max_tokens: Optional[int] = 128          # 最大生成token数，默认128
    temperature: Optional[float] = 0.7       # 温度控制随机性，默认0.7
    top_p: Optional[float] = 0.9             # Top-p采样，默认0.9
    stop: Optional[List[str]] = None         # 停止词列表，默认无

class GenerateResponse(BaseModel):
    text: str                                # 生成文本
    prompt: str                              # 原始提示词
    finish_reason: Optional[str] = None      # 停止原因（"stop"、"length"等）

#----------3.FastAPI生命周期管理----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用启动时初始化LLM模型，关闭时清理资源
    """
    # 启动：初始化LLM
    print("正在初始化LLM模型...")
    global llm
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,             #信任并执行模型仓库中的所有代码
        max_model_len=1024,                 #最大上下文长度
        gpu_memory_utilization=0.5          #GPU显存使用率50%
    )
    print("LLM模型初始化完成")
    
    yield  # 应用在此处正常运行
    
    print("正在关闭LLM模型...")

#----------4.创建FastAPI应用----------
app = FastAPI(
    title="vLLM Inference API",
    description="推理服务",
    lifespan=lifespan
)

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """生成接口"""
    if llm is None:
        raise HTTPException(status_code=503, detail="模型尚未初始化，请稍后重试")
    
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop
    )
    
    # 线程池执行同步llm.generate，避免阻塞事件循环，否则由于llm.generate执行需要时间，整个事件暂停
    import asyncio
    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None,# 默认线程池
        lambda: llm.generate([req.prompt], sampling_params) #传递的函数，lamda匿名函数，用的llm的generate函数
    )
    output = outputs[0]
    generate_text = output.outputs[0].text
    finish_reason = output.outputs[0].finish_reason
    
    return GenerateResponse(
        text=generate_text,    # 生成的结果
        prompt=req.prompt,
        finish_reason=finish_reason   #结束的原因
    )

@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "ok"}

#----------5.启动服务-----------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
