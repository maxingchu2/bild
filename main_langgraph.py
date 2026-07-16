import datetime
import json
import os
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.getenv("UPSTREAM_API_KEY", "sk-e76068d4dd5f4da9a8da51094ff09d91")
MODEL = os.getenv("UPSTREAM_MODEL", "deepseek-v4-flash")

app = FastAPI(title="多轮对话助手（LangGraph 版）")


# ---------- 示例工具：LLM 会根据用户问题自动决定是否调用 ----------

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 "(3 + 5) * 2 / 7"。只支持数字和 + - * / ( ) . 符号。"""
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return "表达式包含不支持的字符"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算出错: {e}"


@tool
def get_current_time() -> str:
    """获取当前的日期和时间（UTC）。"""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def query_employee(name: str) -> str:
    """查询员工信息（示例：模拟查询公司内部数据库/Java 后端接口）。"""
    fake_db = {
        "张三": {"部门": "研发部", "职位": "后端工程师", "工号": "E1001"},
        "李四": {"部门": "产品部", "职位": "产品经理", "工号": "E1002"},
    }
    info = fake_db.get(name)
    return json.dumps(info, ensure_ascii=False) if info else f"未找到员工 {name}"


TOOLS = [calculator, get_current_time, query_employee]

# ---------- LangGraph 图定义：agent(LLM决策) <-> tools(自动执行) ----------

llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL, streaming=True)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = "你是一个有用的中文助手，回答使用 Markdown 格式。需要计算、查时间或查员工信息时调用相应工具。"


async def agent_node(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile()


# ---------- FastAPI 接口：与原版 /api/chat 契约完全一致 ----------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


def to_lc_messages(messages: list[Message]):
    out = []
    for m in messages:
        if m.role == "assistant":
            out.append(AIMessage(content=m.content))
        elif m.role == "system":
            out.append(SystemMessage(content=m.content))
        else:
            out.append(HumanMessage(content=m.content))
    return out


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def stream_graph(req: ChatRequest) -> AsyncGenerator[str, None]:
    inputs = {"messages": to_lc_messages(req.messages)}
    try:
        async for event in graph.astream_events(inputs, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                reasoning = (chunk.additional_kwargs or {}).get("reasoning_content") or ""
                if reasoning:
                    yield sse({"reasoning": reasoning})
                if chunk.content:
                    yield sse({"content": chunk.content})
            elif kind == "on_tool_start":
                name = event.get("name", "tool")
                args = event["data"].get("input")
                yield sse({"reasoning": f"\n[自动执行工具 {name}，参数 {json.dumps(args, ensure_ascii=False)}]\n"})
            elif kind == "on_tool_end":
                name = event.get("name", "tool")
                output = event["data"].get("output")
                text = getattr(output, "content", output)
                yield sse({"reasoning": f"[工具 {name} 返回: {text}]\n"})
    except Exception as e:
        yield sse({"error": f"执行出错: {e}"})
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        stream_graph(req),
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
