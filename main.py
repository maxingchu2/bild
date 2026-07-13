import json
import os
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.deepseek.com/v1/chat/completions")
API_KEY = os.getenv("UPSTREAM_API_KEY", "sk-e76068d4dd5f4da9a8da51094ff09d91")
MODEL = os.getenv("UPSTREAM_MODEL", "deepseek-v4-flash")

app = FastAPI(title="多轮对话助手")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


async def stream_upstream(req: ChatRequest) -> AsyncGenerator[str, None]:
    payload = {
        "model": MODEL,
        "messages": [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            async with client.stream("POST", UPSTREAM_URL, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    err = body.decode("utf-8", errors="replace")
                    yield f"data: {json.dumps({'error': f'上游返回 {resp.status_code}: {err}'}, ensure_ascii=False)}\n\n"
                    return
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        d = chunk["choices"][0]["delta"]
                        delta = d.get("content") or ""
                        reasoning = d.get("reasoning_content") or ""
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if reasoning:
                        yield f"data: {json.dumps({'reasoning': reasoning}, ensure_ascii=False)}\n\n"
                    if delta:
                        yield f"data: {json.dumps({'content': delta}, ensure_ascii=False)}\n\n"
    except httpx.HTTPError as e:
        yield f"data: {json.dumps({'error': f'请求上游失败: {e}'}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        stream_upstream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
