"""船舶检验多智能体服务（LangGraph Supervisor 模式）。

图结构:
    START -> supervisor(路由决策)
             ├─> task_agent(任务查询)      <-> task_tools
             ├─> prepare_agent(检验前准备)  <-> prepare_tools
             ├─> archive_agent(记录归档)    <-> archive_tools
             ├─> report_agent(报告写作)
             └─> general_agent(通用问答)
    各智能体完成后 -> END

对外接口与原版 /api/chat 契约完全一致（SSE: reasoning/content/error/[DONE]），
前端 static/index.html 零改动；智能体切换与工具执行过程实时显示在思考区。
"""

import json
import os
from typing import AsyncGenerator, Literal

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

app = FastAPI(title="船舶检验多智能体")

llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL, streaming=True)


# ==================== 模拟业务数据（对接时替换为真实数据库/Java 接口） ====================

SHIPS = {
    "远洋之星": {
        "CCSNO": "CCS2023001", "船舶类型": "散货船", "建造日期": "2018-05-20",
        "检验类型": "年度检验", "状态": "待检验前准备",
        "检验项": [
            {"编号": 1, "名称": "救生艇及降落装置", "类别": "救生设备"},
            {"编号": 2, "名称": "救生圈及自亮灯", "类别": "救生设备"},
            {"编号": 3, "名称": "应急消防泵压力检查", "类别": "消防设备"},
            {"编号": 4, "名称": "火灾探测器功能试验", "类别": "消防设备"},
            {"编号": 5, "名称": "外板腐蚀测厚", "类别": "外板测厚"},
        ],
        "遗留复查项": [
            {"编号": "L1", "问题": "应急消防泵压力不足", "状态": "未确认"},
            {"编号": "L2", "问题": "船体板厚局部减薄", "状态": "未确认"},
        ],
    },
    "东海明珠": {
        "CCSNO": "CCS2023002", "船舶类型": "集装箱船", "建造日期": "2020-11-02",
        "检验类型": "特别检验", "状态": "待归档",
        "检验项": [
            {"编号": 1, "名称": "主机运行状态检查", "类别": "主机系统"},
            {"编号": 2, "名称": "舵机密封检查", "类别": "舵机系统"},
        ],
        "遗留复查项": [],
    },
}


# ==================== 各智能体的工具 ====================

@tool
def query_pending_tasks() -> str:
    """查询所有待办检验任务（待检验前准备、待归档的船舶列表）。"""
    tasks = [
        {"船名": name, "CCSNO": s["CCSNO"], "检验类型": s["检验类型"], "状态": s["状态"]}
        for name, s in SHIPS.items()
    ]
    return json.dumps(tasks, ensure_ascii=False)


@tool
def query_ship_info(ship_name: str) -> str:
    """根据船名查询船舶基础信息、检验项和遗留复查项。"""
    ship = SHIPS.get(ship_name)
    return json.dumps(ship, ensure_ascii=False) if ship else f"未找到船舶「{ship_name}」"


@tool
def add_inspection_item(ship_name: str, item_name: str, category: str) -> str:
    """为指定船舶新增一个检验项。category 为类别，如 救生设备/消防设备/外板测厚。"""
    ship = SHIPS.get(ship_name)
    if not ship:
        return f"未找到船舶「{ship_name}」"
    items = ship["检验项"]
    if any(i["名称"] == item_name for i in items):
        return f"「{item_name}」已在当前检验项中"
    items.append({"编号": len(items) + 1, "名称": item_name, "类别": category})
    return f"已新增检验项「{item_name}」（{category}），当前共 {len(items)} 项"


@tool
def remove_inspection_item(ship_name: str, item_name: str) -> str:
    """删除指定船舶的某个检验项（删除痕迹保留）。"""
    ship = SHIPS.get(ship_name)
    if not ship:
        return f"未找到船舶「{ship_name}」"
    items = ship["检验项"]
    for i in items:
        if i["名称"] == item_name:
            items.remove(i)
            return f"已删除检验项「{item_name}」（删除痕迹已保留），当前共 {len(items)} 项"
    return f"检验项「{item_name}」不存在"


@tool
def generate_preparation_sheet(ship_name: str) -> str:
    """生成指定船舶的检验前准备单（含验船师版与船东版）。"""
    ship = SHIPS.get(ship_name)
    if not ship:
        return f"未找到船舶「{ship_name}」"
    return json.dumps({
        "准备单": f"{ship_name} {ship['检验类型']}检验前准备单",
        "船舶信息": {k: ship[k] for k in ("CCSNO", "船舶类型", "建造日期")},
        "检验项": ship["检验项"],
        "遗留复查项": ship["遗留复查项"],
        "版本": ["验船师版", "船东版"],
        "状态": "已生成，可预览并推送至船东移动端",
    }, ensure_ascii=False)


@tool
def confirm_legacy_issue(ship_name: str, issue_id: str) -> str:
    """确认指定船舶的遗留复查项（须逐条确认后方可归档）。issue_id 如 L1。"""
    ship = SHIPS.get(ship_name)
    if not ship:
        return f"未找到船舶「{ship_name}」"
    for issue in ship["遗留复查项"]:
        if issue["编号"] == issue_id:
            issue["状态"] = "已确认"
            return f"遗留复查项 {issue_id}「{issue['问题']}」已确认"
    return f"未找到遗留复查项 {issue_id}"


@tool
def archive_record(ship_name: str) -> str:
    """归档指定船舶的检验记录。所有遗留复查项须已确认，否则归档失败。"""
    ship = SHIPS.get(ship_name)
    if not ship:
        return f"未找到船舶「{ship_name}」"
    unconfirmed = [i for i in ship["遗留复查项"] if i["状态"] != "已确认"]
    if unconfirmed:
        return f"归档失败：还有 {len(unconfirmed)} 项遗留复查项未确认: " + json.dumps(unconfirmed, ensure_ascii=False)
    ship["状态"] = "已归档"
    return f"「{ship_name}」检验记录已归档"


# ==================== 多智能体定义 ====================

AGENTS = {
    "task_agent": {
        "描述": "任务查询智能体：查询待办检验任务、船舶基础信息",
        "prompt": "你是船舶检验任务查询智能体。使用工具查询待办任务和船舶信息，用 Markdown 表格清晰呈现结果。",
        "tools": [query_pending_tasks, query_ship_info],
    },
    "prepare_agent": {
        "描述": "检验前准备智能体：新增/删除检验项、生成检验前准备单",
        "prompt": "你是检验前准备智能体。可查询船舶信息、新增或删除检验项、生成检验前准备单。操作完成后总结当前检验项清单。",
        "tools": [query_ship_info, add_inspection_item, remove_inspection_item, generate_preparation_sheet],
    },
    "archive_agent": {
        "描述": "记录归档智能体：确认遗留复查项、归档检验记录",
        "prompt": "你是记录归档智能体。归档前须确认所有遗留复查项。按用户要求确认遗留项并执行归档，报告结果。",
        "tools": [query_ship_info, confirm_legacy_issue, archive_record],
    },
    "report_agent": {
        "描述": "报告写作智能体：撰写检验报告、处置建议等专业文书",
        "prompt": "你是船舶检验报告写作智能体（高级验船师水平）。根据用户要求撰写检验报告、缺陷处置建议等，使用规范的 Markdown 文书格式，符合中国船级社(CCS)规范表述。",
        "tools": [query_ship_info],
    },
    "general_agent": {
        "描述": "通用问答智能体：船检法规、常见问题及其他通用问题",
        "prompt": "你是船舶检验领域的通用问答助手，熟悉 CCS 规范、SOLAS 公约等，用 Markdown 回答用户问题。",
        "tools": [],
    },
}


class AgentState(MessagesState):
    next_agent: str


ROUTER_PROMPT = (
    "你是船舶检验多智能体系统的调度员(supervisor)。根据用户最新需求，从下列智能体中选择最合适的一个：\n"
    + "\n".join(f"- {name}: {cfg['描述']}" for name, cfg in AGENTS.items())
    + "\n只输出智能体名称。"
)


class Route(BaseModel):
    next_agent: Literal["task_agent", "prepare_agent", "archive_agent", "report_agent", "general_agent"] = Field(
        description="要调度的智能体名称"
    )


router_llm = llm.with_structured_output(Route)


async def supervisor_node(state: AgentState):
    messages = [SystemMessage(content=ROUTER_PROMPT)] + state["messages"]
    route = await router_llm.ainvoke(messages)
    return {"next_agent": route.next_agent}


def make_agent_node(name: str):
    cfg = AGENTS[name]
    agent_llm = llm.bind_tools(cfg["tools"]) if cfg["tools"] else llm

    async def node(state: AgentState):
        messages = [SystemMessage(content=cfg["prompt"])] + state["messages"]
        response = await agent_llm.ainvoke(messages)
        return {"messages": [response]}

    node.__name__ = name
    return node


def make_should_continue(tools_node_name: str):
    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return tools_node_name
        return END

    return should_continue


builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", lambda s: s["next_agent"], list(AGENTS.keys()))

for name, cfg in AGENTS.items():
    builder.add_node(name, make_agent_node(name))
    if cfg["tools"]:
        tools_name = f"{name}_tools"
        builder.add_node(tools_name, ToolNode(cfg["tools"]))
        builder.add_conditional_edges(name, make_should_continue(tools_name), [tools_name, END])
        builder.add_edge(tools_name, name)
    else:
        builder.add_edge(name, END)

graph = builder.compile()


# ==================== FastAPI 接口（契约与原版一致） ====================

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
            if kind == "on_chain_start" and event.get("name") in AGENTS:
                desc = AGENTS[event["name"]]["描述"]
                yield sse({"reasoning": f"\n[调度至 {desc}]\n"})
            elif kind == "on_chat_model_stream":
                if event.get("metadata", {}).get("langgraph_node") == "supervisor":
                    continue
                chunk = event["data"]["chunk"]
                reasoning = (chunk.additional_kwargs or {}).get("reasoning_content") or ""
                if reasoning:
                    yield sse({"reasoning": reasoning})
                if chunk.content:
                    yield sse({"content": chunk.content})
            elif kind == "on_tool_start":
                name = event.get("name", "tool")
                args = event["data"].get("input")
                yield sse({"reasoning": f"\n[执行工具 {name}，参数 {json.dumps(args, ensure_ascii=False)}]\n"})
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


@app.get("/api/agents")
async def list_agents():
    return {name: cfg["描述"] for name, cfg in AGENTS.items()}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
