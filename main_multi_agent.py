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

import csv
import json
import os
import uuid
from datetime import datetime
from typing import AsyncGenerator, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.getenv("UPSTREAM_API_KEY", "sk-e76068d4dd5f4da9a8da51094ff09d91")
MODEL = os.getenv("UPSTREAM_MODEL", "deepseek-v4-flash")

app = FastAPI(title="船舶检验多智能体")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL, streaming=True)


# ==================== 业务数据：CSV 文件即数据接口（对接时替换为真实数据库/Java 接口） ====================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHIPS_CSV = os.path.join(DATA_DIR, "ships.csv")
ITEMS_CSV = os.path.join(DATA_DIR, "inspection_items.csv")
ISSUES_CSV = os.path.join(DATA_DIR, "legacy_issues.csv")
CONVS_CSV = os.path.join(DATA_DIR, "conversations.csv")


def load_ships() -> dict:
    ships = {}
    with open(SHIPS_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.pop("船名")
            ships[name] = {**row, "检验项": [], "遗留复查项": []}
    with open(ITEMS_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["船名"] in ships:
                ships[row["船名"]]["检验项"].append(
                    {"编号": row["编号"], "名称": row["名称"], "类别": row["类别"]}
                )
    with open(ISSUES_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["船名"] in ships:
                ships[row["船名"]]["遗留复查项"].append(
                    {"编号": row["编号"], "问题": row["问题"], "状态": row["状态"]}
                )
    return ships


def save_ships() -> None:
    with open(SHIPS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["船名", "CCSNO", "船舶类型", "建造日期", "检验类型", "状态"])
        for name, s in SHIPS.items():
            w.writerow([name, s["CCSNO"], s["船舶类型"], s["建造日期"], s["检验类型"], s["状态"]])
    with open(ITEMS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["船名", "编号", "名称", "类别"])
        for name, s in SHIPS.items():
            for i in s["检验项"]:
                w.writerow([name, i["编号"], i["名称"], i["类别"]])
    with open(ISSUES_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["船名", "编号", "问题", "状态"])
        for name, s in SHIPS.items():
            for i in s["遗留复查项"]:
                w.writerow([name, i["编号"], i["问题"], i["状态"]])


SHIPS = load_ships()


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
def add_inspection_item(ship_name: str, item_name: str, category: str, item_id: str = "") -> str:
    """为指定船舶新增一个检验项。category 为类别，如 救生设备/消防设备/外板测厚；item_id 可选，为检验项编号（如 FC-119），不填则自动编号。"""
    ship = SHIPS.get(ship_name)
    if not ship:
        return f"未找到船舶「{ship_name}」"
    items = ship["检验项"]
    if any(i["名称"] == item_name for i in items):
        return f"「{item_name}」已在当前检验项中"
    if item_id and any(i["编号"] == item_id for i in items):
        exist = next(i for i in items if i["编号"] == item_id)
        return f"新增失败：编号「{item_id}」已被检验项「{exist['名称']}」占用，请更换编号或不指定编号（自动编号）"
    new_id = item_id or f"NEW-{len(items) + 1}"
    items.append({"编号": new_id, "名称": item_name, "类别": category})
    save_ships()
    return f"已新增检验项 {new_id}「{item_name}」（{category}），已写入 CSV，当前共 {len(items)} 项"


@tool
def remove_inspection_item(ship_name: str, item_name: str) -> str:
    """删除指定船舶的某个检验项，item_name 可以是检验项名称或编号（如 FC-125）。"""
    ship = SHIPS.get(ship_name)
    if not ship:
        return f"未找到船舶「{ship_name}」"
    items = ship["检验项"]
    for i in items:
        if item_name in (i["名称"], i["编号"]):
            items.remove(i)
            save_ships()
            return f"已删除检验项 {i['编号']}「{i['名称']}」（已写入 CSV），当前共 {len(items)} 项"
    return f"删除失败：检验项「{item_name}」不存在（可用名称或编号删除），请先查询船舶信息确认检验项清单"


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
            save_ships()
            return f"遗留复查项 {issue_id}「{issue['问题']}」已确认（已写入 CSV）"
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
    save_ships()
    return f"「{ship_name}」检验记录已归档（已写入 CSV）"


# ==================== 多智能体定义 ====================

PLAIN_STYLE = (
    "输出格式要求：使用规范的 Markdown 排版，清单/数据用表格呈现，小节用标题，"
    "重点内容加粗，可适当使用 emoji 图标，保持美观易读。"
)

AGENTS = {
    "task_agent": {
        "描述": "任务查询智能体：查询待办检验任务、船舶基础信息",
        "prompt": (
            "你是船舶检验任务查询智能体。使用工具查询待办任务和船舶信息，逐行清晰呈现结果。"
            "当用户开始新的会话或请求引导时，先调用 query_pending_tasks 展示待办任务，"
            "然后引导用户选择下一步（如：查看某船详情、开始检验前准备、确认遗留项并归档、撰写报告）。"
            "每次回答末尾给出可选的下一步操作建议，逐步引导用户完成检验流程。"
            + PLAIN_STYLE
        ),
        "tools": [query_pending_tasks, query_ship_info],
    },
    "prepare_agent": {
        "描述": "检验前准备智能体：新增/删除检验项、生成检验前准备单",
        "prompt": "你是检验前准备智能体。可查询船舶信息、新增或删除检验项、生成检验前准备单。操作完成后总结当前检验项清单。" + PLAIN_STYLE,
        "tools": [query_ship_info, add_inspection_item, remove_inspection_item, generate_preparation_sheet],
    },
    "archive_agent": {
        "描述": "记录归档智能体：确认遗留复查项、归档检验记录",
        "prompt": "你是记录归档智能体。归档前须确认所有遗留复查项。按用户要求确认遗留项并执行归档，报告结果。" + PLAIN_STYLE,
        "tools": [query_ship_info, confirm_legacy_issue, archive_record],
    },
    "report_agent": {
        "描述": "报告写作智能体：撰写检验报告、处置建议等专业文书",
        "prompt": "你是船舶检验报告写作智能体（高级验船师水平）。根据用户要求撰写检验报告、缺陷处置建议等，符合中国船级社(CCS)规范表述。文书用纯文本排版（标题、章节编号、空行分段）。" + PLAIN_STYLE,
        "tools": [query_ship_info],
    },
    "general_agent": {
        "描述": "通用问答智能体：船检法规、常见问题及其他通用问题",
        "prompt": "你是船舶检验领域的通用问答助手，熟悉 CCS 规范、SOLAS 公约等，简洁准确地回答用户问题。" + PLAIN_STYLE,
        "tools": [],
    },
}


class AgentState(MessagesState):
    next_agent: str


ROUTER_PROMPT = (
    "你是船舶检验多智能体系统的调度员(supervisor)。根据用户最新需求，从下列智能体中选择最合适的一个：\n"
    + "\n".join(f"- {name}: {cfg['描述']}" for name, cfg in AGENTS.items())
    + "\n路由参考：开始新会话/请求引导/查询待办任务/船舶信息→task_agent；新增/删除检验项、生成检验前准备单→prepare_agent；"
    "确认遗留项/归档→archive_agent；撰写检验报告/文书→report_agent；其他法规咨询等→general_agent。"
    "\n只输出一行英文智能体名称（如 task_agent），不要输出其他内容。"
)


class Route(BaseModel):
    next_agent: Literal["task_agent", "prepare_agent", "archive_agent", "report_agent", "general_agent"] = Field(
        description="要调度的智能体名称"
    )


async def supervisor_node(state: AgentState):
    messages = [SystemMessage(content=ROUTER_PROMPT)] + state["messages"]
    response = await llm.ainvoke(messages)
    text = response.content if isinstance(response.content, str) else str(response.content)
    next_agent = "general_agent"
    for name, cfg in AGENTS.items():
        if name in text or cfg["描述"].split("：")[0] in text:
            next_agent = name
            break
    return {"next_agent": next_agent}


def make_agent_node(name: str):
    cfg = AGENTS[name]
    agent_llm = llm.bind_tools(cfg["tools"]) if cfg["tools"] else llm

    async def node(state: AgentState):
        messages = [SystemMessage(content=cfg["prompt"])] + state["messages"]
        response = None
        async for chunk in agent_llm.astream(messages):
            response = chunk if response is None else response + chunk
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
    current_agent = None
    try:
        async for msg, meta in graph.astream(inputs, stream_mode="messages"):
            node = meta.get("langgraph_node")
            if node == "supervisor":
                continue
            if node in AGENTS and node != current_agent:
                current_agent = node
                yield sse({"reasoning": f"\n[调度至 {AGENTS[node]['描述']}]\n"})
            if isinstance(msg, ToolMessage):
                yield sse({"reasoning": f"[工具 {msg.name} 返回: {msg.content}]\n"})
                continue
            reasoning = (msg.additional_kwargs or {}).get("reasoning_content") or ""
            if reasoning:
                yield sse({"reasoning": reasoning})
            for tc in msg.tool_calls or []:
                if tc.get("name"):
                    yield sse({"reasoning": f"\n[执行工具 {tc['name']}，参数 {json.dumps(tc.get('args'), ensure_ascii=False)}]\n"})
            if msg.content:
                yield sse({"content": msg.content})
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


# ==================== 页面数据接口：任务概览 / 检验项写回 ====================

class ItemPayload(BaseModel):
    编号: str = ""
    名称: str
    类别: str = ""


class SaveItemsRequest(BaseModel):
    items: list[ItemPayload]


class SetStatusRequest(BaseModel):
    状态: str


@app.get("/api/overview")
async def overview():
    counts = {"待检验": 0, "检验中": 0, "记录归档": 0, "已完成": 0}
    for s in SHIPS.values():
        st = s["状态"]
        if st == "待检验前准备":
            counts["待检验"] += 1
        elif st == "检验中":
            counts["检验中"] += 1
        elif st == "待归档":
            counts["记录归档"] += 1
        else:
            counts["已完成"] += 1
    return counts


@app.get("/api/ships")
async def list_ships():
    return SHIPS


@app.post("/api/ships/{ship_name}/items")
async def save_ship_items(ship_name: str, req: SaveItemsRequest):
    ship = SHIPS.get(ship_name)
    if not ship:
        return {"saved": False, "error": f"未找到船舶「{ship_name}」"}
    ship["检验项"] = [
        {"编号": it.编号 or f"NEW-{idx + 1}", "名称": it.名称, "类别": it.类别}
        for idx, it in enumerate(req.items)
    ]
    save_ships()
    return {"saved": True, "count": len(ship["检验项"])}


@app.post("/api/ships/{ship_name}/status")
async def set_ship_status(ship_name: str, req: SetStatusRequest):
    ship = SHIPS.get(ship_name)
    if not ship:
        return {"saved": False, "error": f"未找到船舶「{ship_name}」"}
    ship["状态"] = req.状态
    save_ships()
    return {"saved": True, "状态": req.状态}


# ==================== 历史对话：CSV 持久化 ====================

CONV_FIELDS = ["id", "title", "time", "messages"]


def load_conversations() -> list[dict]:
    if not os.path.exists(CONVS_CSV):
        return []
    with open(CONVS_CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_conversations(convs: list[dict]) -> None:
    with open(CONVS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONV_FIELDS)
        w.writeheader()
        w.writerows(convs)


class SaveConversationRequest(BaseModel):
    messages: list[Message]
    title: str = ""


@app.get("/api/conversations")
async def list_conversations():
    return [
        {"id": c["id"], "title": c["title"], "time": c["time"]}
        for c in reversed(load_conversations())
    ]


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    for c in load_conversations():
        if c["id"] == conv_id:
            return {
                "id": c["id"],
                "title": c["title"],
                "time": c["time"],
                "messages": json.loads(c["messages"]),
            }
    return {"id": conv_id, "title": "", "time": "", "messages": []}


@app.post("/api/conversations")
async def save_conversation(req: SaveConversationRequest):
    if not req.messages:
        return {"saved": False}
    title = req.title.strip()
    if not title:
        first_user = next((m.content for m in req.messages if m.role == "user"), "新对话")
        if first_user.startswith("开始新的检验工作会话"):
            first_user = "检验工作会话"
        title = first_user.replace("\n", " ")[:24]
    conv = {
        "id": uuid.uuid4().hex,
        "title": title,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": json.dumps(
            [{"role": m.role, "content": m.content} for m in req.messages],
            ensure_ascii=False,
        ),
    }
    convs = load_conversations()
    convs.append(conv)
    save_conversations(convs)
    return {"saved": True, "id": conv["id"], "title": conv["title"], "time": conv["time"]}


@app.get("/ship", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
@app.get("/assistant", response_class=HTMLResponse)
@app.get("/preparations/{_path:path}", response_class=HTMLResponse)
@app.get("/archives/{_path:path}", response_class=HTMLResponse)
async def ship_page(_path: str = ""):
    html_path = os.path.join(os.path.dirname(__file__), "static", "ship-inspection.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-cache"})


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
