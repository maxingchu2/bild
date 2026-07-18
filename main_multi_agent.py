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
import re
import uuid
from datetime import datetime
from typing import AsyncGenerator, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    print(
        f"[422] path={request.url.path} errors={exc.errors()} body={body.decode('utf-8', 'replace')[:2000]}",
        flush=True,
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


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
                    {"编号": row["编号"], "名称": row["名称"], "类别": row["类别"], "风险": row.get("风险") or "低风险"}
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
        w.writerow(["船名", "编号", "名称", "类别", "风险"])
        for name, s in SHIPS.items():
            for i in s["检验项"]:
                w.writerow([name, i["编号"], i["名称"], i["类别"], i.get("风险", "低风险")])
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
def add_inspection_item(ship_name: str, item_name: str, category: str, item_id: str = "", risk: str = "低风险") -> str:
    """为指定船舶新增一个检验项。category 为类别，如 救生设备/消防设备/外板测厚；item_id 可选，为检验项编号（如 FC-119），不填则自动编号；risk 可选，为风险等级（高风险/中风险/低风险，默认低风险）。"""
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
    if risk not in ("高风险", "中风险", "低风险"):
        risk = "低风险"
    items.append({"编号": new_id, "名称": item_name, "类别": category, "风险": risk})
    save_ships()
    return f"已新增检验项 {new_id}「{item_name}」（{category}，{risk}），已写入 CSV，当前共 {len(items)} 项"


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


# ==================== 船检智能体模型对话流式接口（/api/ai/chat/classify） ====================

TODO_BACKEND_BASE = os.getenv("TODO_BACKEND_BASE_URL", "http://5.5.5.45:8082")

STATUS_CODE_MAP = {
    "待检验前准备": ("pending_preparation", "待准备"),
    "待检验": ("pending_inspection", "待检验"),
    "检验中": ("inspection", "检验中"),
    "待归档": ("archive", "待归档"),
}


class ClassifyRequest(BaseModel):
    model_config = {"extra": "ignore"}

    sessionId: int | str | None = None
    taskId: int | str | None = None
    clientType: str | None = "pc"
    pageCode: str | None = "home"
    sessionType: str | None = "mixed"
    clientMessageId: str | None = ""
    content: str = ""
    actionCode: str | None = ""
    actionParams: dict | None = None
    attachmentIds: list[int | str] | None = None


def sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    print(f"[SSE] event={event} data={payload}", flush=True)
    return f"event: {event}\ndata: {payload}\n\n"


def local_todo_summary(page: int, limit: int) -> dict:
    """后端接口不可达时的降级数据源：使用本地船舶任务数据统计待办。"""
    status_counts: dict = {}
    todo_list = []
    task_id = 0
    for name, s in SHIPS.items():
        task_id += 1
        mapped = STATUS_CODE_MAP.get(s["状态"])
        if not mapped:
            continue  # 已归档/已完成不计入待办
        code, code_name = mapped
        status_counts[code] = status_counts.get(code, 0) + 1
        todo_list.append(
            {
                "taskId": task_id,
                "taskNo": f"LOCAL-TASK-{task_id:04d}",
                "shipId": task_id,
                "shipName": name,
                "ccsNo": s.get("CCSNO", ""),
                "inspectionType": "annual" if s.get("检验类型") == "年度检验" else "special",
                "inspectionTypeName": s.get("检验类型", ""),
                "status": code,
                "statusName": code_name,
                "plannedInspectionDate": "",
                "surveyorName": "张工",
                "checkItemCount": len(s.get("检验项", [])),
                "legacyItemCount": len(s.get("遗留复查项", [])),
                "issueCount": len(
                    [x for x in s.get("遗留复查项", []) if x.get("状态") == "未确认"]
                ),
                "progressPercent": 0,
            }
        )
    total = len(todo_list)
    start = (page - 1) * limit
    return {
        "todoTotal": total,
        "statusCounts": status_counts,
        "todoList": todo_list[start : start + limit],
        "source": "local_ships",
    }


async def fetch_todo_summary(
    client_type: str, page: int, limit: int, auth: str
) -> dict:
    """优先调用 5.5.5.45:8082 的任务列表/工作台汇总接口，失败时降级为本地数据。"""
    headers = {"Accept": "application/json"}
    if auth:
        headers["Authorization"] = auth
    async with httpx.AsyncClient(base_url=TODO_BACKEND_BASE, timeout=5) as client:
        tasks_resp = await client.get(
            "/api/ai/ship-tasks",
            params={
                "clientType": client_type,
                "page": page,
                "limit": limit,
                "todoFilter": "all",
            },
            headers=headers,
        )
        tasks_resp.raise_for_status()
        tasks_data = tasks_resp.json().get("data") or {}
        result = {
            "todoTotal": tasks_data.get("total"),
            "statusCounts": {},
            "todoList": tasks_data.get("list") or [],
            "source": "ship_tasks",
        }
        try:
            summary_resp = await client.get(
                "/api/ai/workbench/summary",
                params={"clientType": client_type},
                headers=headers,
            )
            summary_resp.raise_for_status()
            summary_data = summary_resp.json().get("data") or {}
            result["statusCounts"] = summary_data.get("statusCounts") or {}
            if result["todoTotal"] is None:
                result["todoTotal"] = summary_data.get("todoTotal")
                result["source"] = "workbench_summary"
        except Exception as e:
            print(f"[todo] 工作台汇总接口不可用: {e}", flush=True)
        return result


async def stream_classify(req: ClassifyRequest, auth: str) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    session_id = req.sessionId or int(datetime.now().timestamp() * 1000)
    if isinstance(session_id, str) and session_id.isdigit():
        session_id = int(session_id)
    turn_id = int(datetime.now().timestamp() * 1000) + 1
    user_message_id = turn_id + 1
    seq = 0

    yield sse_event(
        "message_start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": None,
            "status": "running",
        },
    )

    params = req.actionParams or {}
    page = int(params.get("todoPage") or 1)
    limit = int(params.get("todoLimit") or 20)
    try:
        todo = await fetch_todo_summary(req.clientType or "pc", page, limit, auth)
    except Exception as e:
        print(f"[todo] 后端待办接口不可达（{TODO_BACKEND_BASE}），降级为本地数据: {e}", flush=True)
        try:
            todo = local_todo_summary(page, limit)
        except Exception as e2:
            print(f"[todo] 本地待办统计失败: {e2}", flush=True)
            todo = None

    seq += 1
    if todo is None:
        yield sse_event(
            "answer_delta",
            {
                "seq": seq,
                "type": "todo_summary",
                "actionCode": "QUERY_MY_TODOS",
                "actionName": "查询我的待办",
                "content": "暂时无法获取待办统计，我继续为你处理问题。",
                "todoTotal": None,
                "todoPage": page,
                "todoLimit": limit,
                "statusCounts": {},
                "todoList": [],
                "source": "",
            },
        )
    else:
        total = todo["todoTotal"]
        shown = len(todo["todoList"])
        content = (
            f"根据当前任务统计，你当前共有 {total} 项待办任务"
            + (f"，下面是优先展示的 {shown} 条待办。" if shown else "。")
            if total is not None
            else "暂时无法获取待办统计，我继续为你处理问题。"
        )
        yield sse_event(
            "answer_delta",
            {
                "seq": seq,
                "type": "todo_summary",
                "actionCode": "QUERY_MY_TODOS",
                "actionName": "查询我的待办",
                "content": content,
                "todoTotal": total,
                "todoPage": page,
                "todoLimit": limit,
                "statusCounts": todo["statusCounts"],
                "todoList": todo["todoList"],
                "source": todo["source"],
            },
        )

    if req.actionCode:
        yield sse_event(
            "action_intent",
            {"actionCode": req.actionCode, "actionName": "查询我的待办" if req.actionCode == "QUERY_MY_TODOS" else req.actionCode},
        )
        if req.actionCode == "QUERY_MY_TODOS" and todo is not None:
            yield sse_event(
                "action_result",
                {
                    "actionCode": "QUERY_MY_TODOS",
                    "status": "success",
                    "message": "查询成功",
                    "payload": {
                        "todoTotal": todo["todoTotal"],
                        "todoList": todo["todoList"],
                    },
                },
            )

    status = "success"
    try:
        inputs = {"messages": [HumanMessage(content=req.content or "查询我的待办任务")]}
        async for msg, meta in graph.astream(inputs, stream_mode="messages"):
            node = meta.get("langgraph_node")
            if node == "supervisor" or isinstance(msg, ToolMessage):
                continue
            if msg.content:
                seq += 1
                yield sse_event(
                    "answer_delta",
                    {"seq": seq, "type": "model_delta", "content": msg.content},
                )
    except Exception as e:
        status = "error"
        yield sse_event("error", {"requestId": request_id, "message": f"模型调用失败: {e}"})

    if status == "success":
        yield sse_event(
            "message_end",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "turnId": turn_id,
                "userMessageId": user_message_id,
                "assistantMessageId": user_message_id + 1,
                "status": "success",
                "actionCode": req.actionCode or "GENERAL_QA",
            },
        )


# ==================== 新增检验项（ADD_CHECK_ITEM） ====================

RISK_LEVEL_NAMES = {"high": "高风险", "medium": "中风险", "low": "低风险"}
RISK_WORD_MAP = {"高": "high", "中": "medium", "低": "low"}
ADD_CHECK_ITEM_PATTERN = re.compile(r"新增检[验查]项")

LOCAL_CHECK_ITEMS: dict = {}
LOCAL_ITEM_ID_SEQ = {"value": 90000}


def detect_add_check_item(req: ClassifyRequest) -> bool:
    if req.actionCode == "ADD_CHECK_ITEM":
        return True
    return bool(req.content and ADD_CHECK_ITEM_PATTERN.search(req.content))


def parse_check_item_params(content: str, params: dict | None) -> dict:
    out = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    text = content or ""
    if not out.get("itemName"):
        m = re.search(r"新增检[验查]项[：:，,\s]*([^，,。;；\s]+)", text)
        if m:
            out["itemName"] = m.group(1)
    if not out.get("itemCode"):
        m = re.search(r"编号[：:\s]*([A-Za-z][A-Za-z0-9\-_]*)", text) or re.search(
            r"\b([A-Z]{2,}-\d+)\b", text
        )
        if m:
            out["itemCode"] = m.group(1)
    if not out.get("disciplineName"):
        m = re.search(r"专业[：:\s]*([^，,。;；\s]+)", text)
        if m:
            out["disciplineName"] = m.group(1)
    if not out.get("sectionName"):
        m = re.search(r"分组[：:\s]*([^，,。;；\s]+)", text)
        if m:
            out["sectionName"] = m.group(1)
    if not out.get("riskLevel"):
        m = re.search(r"风险等级[：:\s]*(高|中|低|high|medium|low)", text)
        if m:
            out["riskLevel"] = RISK_WORD_MAP.get(m.group(1), m.group(1))
    return out


def count_check_items(tree: list) -> int:
    total = 0
    for discipline in tree or []:
        count = discipline.get("count")
        if count is not None:
            total += int(count)
        else:
            for section in discipline.get("sections") or []:
                total += len(section.get("items") or [])
    return total


def backend_headers(auth: str) -> dict:
    headers = {"Accept": "application/json"}
    if auth:
        headers["Authorization"] = auth
    return headers


async def add_check_item_backend(task_id, body: dict, auth: str) -> dict:
    async with httpx.AsyncClient(base_url=TODO_BACKEND_BASE, timeout=10) as client:
        resp = await client.post(
            f"/api/ai/ship-tasks/{task_id}/check-items",
            json=body,
            headers=backend_headers(auth),
        )
        resp.raise_for_status()
        return (resp.json() or {}).get("data") or {}


async def fetch_check_item_tree(task_id, auth: str) -> list:
    async with httpx.AsyncClient(base_url=TODO_BACKEND_BASE, timeout=10) as client:
        resp = await client.get(
            f"/api/ai/ship-tasks/{task_id}/check-items",
            params={"includeDeleted": "false"},
            headers=backend_headers(auth),
        )
        resp.raise_for_status()
        data = (resp.json() or {}).get("data")
        return data or []


def local_add_check_item(task_id, body: dict) -> dict:
    """后端不可达时的降级：在本地内存中维护该任务的检查项。"""
    LOCAL_ITEM_ID_SEQ["value"] += 1
    item = {
        "itemId": LOCAL_ITEM_ID_SEQ["value"],
        "itemCode": body.get("itemCode", ""),
        "itemName": body.get("itemName", ""),
        "disciplineCode": body.get("disciplineCode", ""),
        "disciplineName": body.get("disciplineName", "") or "未分类",
        "sectionName": body.get("sectionName", "") or "默认分组",
        "riskLevel": body.get("riskLevel", ""),
        "riskLevelName": RISK_LEVEL_NAMES.get(body.get("riskLevel", ""), ""),
        "status": "pending",
        "regulationBasis": body.get("regulationBasis"),
        "actionRequirement": body.get("actionRequirement"),
    }
    LOCAL_CHECK_ITEMS.setdefault(str(task_id), []).append(item)
    return item


def local_check_item_tree(task_id) -> list:
    disciplines: dict = {}
    for item in LOCAL_CHECK_ITEMS.get(str(task_id), []):
        d_key = (item.get("disciplineCode") or "", item.get("disciplineName") or "未分类")
        discipline = disciplines.setdefault(
            d_key,
            {
                "disciplineCode": d_key[0],
                "disciplineName": d_key[1],
                "count": 0,
                "sections": {},
            },
        )
        discipline["count"] += 1
        section = discipline["sections"].setdefault(
            item.get("sectionName") or "默认分组",
            {"sectionName": item.get("sectionName") or "默认分组", "items": []},
        )
        section["items"].append(
            {
                "itemId": item["itemId"],
                "itemCode": item["itemCode"],
                "itemName": item["itemName"],
                "riskLevel": item["riskLevel"],
                "riskLevelName": item["riskLevelName"],
                "status": item["status"],
                "regulationBasis": item.get("regulationBasis"),
                "actionRequirement": item.get("actionRequirement"),
            }
        )
    return [
        {**d, "sections": list(d["sections"].values())} for d in disciplines.values()
    ]


async def stream_add_check_item(
    req: ClassifyRequest, auth: str
) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    session_id = req.sessionId or int(datetime.now().timestamp() * 1000)
    if isinstance(session_id, str) and session_id.isdigit():
        session_id = int(session_id)
    turn_id = int(datetime.now().timestamp() * 1000) + 1
    user_message_id = turn_id + 1
    seq = 0

    yield sse_event(
        "message_start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": None,
            "status": "running",
        },
    )

    task_id = req.taskId
    if isinstance(task_id, str) and task_id.isdigit():
        task_id = int(task_id)
    params = parse_check_item_params(req.content or "", req.actionParams)

    def base_delta(status: str, content: str) -> dict:
        return {
            "seq": 1,
            "type": "action_result",
            "actionCode": "ADD_CHECK_ITEM",
            "actionName": "新增检查项",
            "status": status,
            "content": content,
            "taskId": task_id,
            "addedItem": None,
            "checkItemTotal": None,
            "checkItemTree": [],
        }

    rejected = None
    if task_id in (None, ""):
        rejected = base_delta("rejected", "新增失败：该操作需要关联检验任务。")
    elif not params.get("itemCode") or not params.get("itemName"):
        rejected = base_delta("rejected", "新增失败：缺少检验项编号或检验项名称。")

    if rejected is not None:
        seq = 1
        yield sse_event("answer_delta", rejected)
        yield sse_event(
            "message_end",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "turnId": turn_id,
                "userMessageId": user_message_id,
                "assistantMessageId": user_message_id + 1,
                "status": "success",
                "actionCode": "ADD_CHECK_ITEM",
            },
        )
        return

    body = {
        "itemCode": params.get("itemCode", ""),
        "itemName": params.get("itemName", ""),
        "disciplineCode": params.get("disciplineCode", ""),
        "disciplineName": params.get("disciplineName", ""),
        "sectionName": params.get("sectionName", ""),
        "riskLevel": params.get("riskLevel", ""),
        "source": "ai",
    }
    for key in ("categoryName", "regulationBasis", "actionRequirement"):
        if params.get(key):
            body[key] = params[key]

    added_item = None
    tree_source = "check_items_after_add"
    try:
        added_item = await add_check_item_backend(task_id, body, auth)
    except Exception as e:
        print(
            f"[check-item] 后端新增接口不可达（{TODO_BACKEND_BASE}），降级为本地数据: {e}",
            flush=True,
        )
        added_item = local_add_check_item(task_id, body)
        tree_source = "local_check_items"

    if not added_item or not added_item.get("itemId"):
        added_item = {
            **(added_item or {}),
            "itemCode": body["itemCode"],
            "itemName": body["itemName"],
            "disciplineCode": body["disciplineCode"],
            "disciplineName": body["disciplineName"],
            "sectionName": body["sectionName"],
            "riskLevel": body["riskLevel"],
            "riskLevelName": RISK_LEVEL_NAMES.get(body["riskLevel"], ""),
            "status": "pending",
        }

    tree = None
    if tree_source == "local_check_items":
        tree = local_check_item_tree(task_id)
    else:
        try:
            tree = await fetch_check_item_tree(task_id, auth)
        except Exception as e:
            print(f"[check-item] 检查项树查询失败: {e}", flush=True)

    seq = 1
    if tree is None:
        delta = base_delta(
            "partial_success", "新增成功，但暂时无法获取最新检查项列表。"
        )
        delta["addedItem"] = added_item
        yield sse_event("answer_delta", delta)
    else:
        total = count_check_items(tree)
        delta = base_delta("success", f"新增成功，现有 {total} 条检查项。")
        delta.update(
            {
                "addedItem": added_item,
                "checkItemTotal": total,
                "checkItemTree": tree,
                "refresh": ["checkItems", "overview"],
                "source": tree_source,
            }
        )
        yield sse_event("answer_delta", delta)
        yield sse_event(
            "action_result",
            {
                "actionCode": "ADD_CHECK_ITEM",
                "status": "success",
                "message": "新增成功",
                "payload": {
                    "taskId": task_id,
                    "addedItem": added_item,
                    "checkItemTotal": total,
                },
            },
        )

    yield sse_event(
        "message_end",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": user_message_id + 1,
            "status": "success",
            "actionCode": "ADD_CHECK_ITEM",
        },
    )


# ==================== 删除检查项（DELETE_CHECK_ITEM） ====================

DELETE_CHECK_ITEM_PATTERN = re.compile(
    r"(删除|移除|去掉).{0,40}检[验查]|检[验查]项.{0,20}(删除|移除|去掉)"
)


def detect_delete_check_item(req: ClassifyRequest) -> bool:
    if req.actionCode == "DELETE_CHECK_ITEM":
        return True
    if req.actionCode:
        return False
    return bool(req.content and DELETE_CHECK_ITEM_PATTERN.search(req.content))


def flatten_check_item_tree(tree: list) -> list:
    flat = []
    for discipline in tree or []:
        for section in discipline.get("sections") or []:
            for item in section.get("items") or []:
                flat.append(
                    {
                        **item,
                        "disciplineCode": discipline.get("disciplineCode"),
                        "disciplineName": discipline.get("disciplineName"),
                        "sectionName": section.get("sectionName"),
                    }
                )
    return flat


def parse_delete_target(content: str, params: dict | None) -> dict:
    out = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    text = content or ""
    if not out.get("itemCode"):
        m = re.search(r"编号[：:\s]*([A-Za-z][A-Za-z0-9\-_]*)", text) or re.search(
            r"\b([A-Z]{2,}-\d+)\b", text
        )
        if m:
            out["itemCode"] = m.group(1)
    if not out.get("targetText") and not out.get("itemName") and not out.get("itemId"):
        m = (
            re.search(r"(?:删除|移除|去掉)[：:，,\s]*(.+?)(?:这个|这条|这一项)?检[验查]项", text)
            or re.search(r"(?:删除|移除|去掉)检[验查]项[：:，,\s]*([^，,。;；\s]+)", text)
            or re.search(r"把(.+?)(?:这个|这条|这一项)?检[验查]项?(?:从.*)?(?:删除|移除|去掉)", text)
        )
        if m:
            target = m.group(1).strip("，, 　")
            if target:
                out["targetText"] = target
    return out


def match_check_items(flat: list, target: dict) -> list:
    item_id = target.get("itemId")
    if item_id not in (None, ""):
        matched = [i for i in flat if str(i.get("itemId")) == str(item_id)]
        if matched:
            return matched

    item_code = target.get("itemCode") or ""
    if item_code:
        exact = [i for i in flat if (i.get("itemCode") or "") == item_code]
        if exact:
            return exact
        partial = [i for i in flat if item_code in (i.get("itemCode") or "")]
        if partial:
            return partial

    item_name = target.get("itemName") or ""
    if item_name:
        exact = [i for i in flat if (i.get("itemName") or "") == item_name]
        if exact:
            return exact
        partial = [i for i in flat if item_name in (i.get("itemName") or "")]
        if partial:
            return partial

    text = target.get("targetText") or ""
    if text:
        exact = [
            i
            for i in flat
            if text in ((i.get("itemName") or ""), (i.get("itemCode") or ""))
        ]
        if exact:
            return exact
        partial = [
            i
            for i in flat
            if text in (i.get("itemName") or "")
            or text in (i.get("itemCode") or "")
            or text in (i.get("sectionName") or "")
            or text in (i.get("disciplineName") or "")
        ]
        if partial:
            return partial

    candidates = flat
    has_filter = False
    for key, field in (
        ("disciplineName", "disciplineName"),
        ("sectionName", "sectionName"),
        ("riskLevel", "riskLevel"),
    ):
        value = target.get(key)
        if value:
            has_filter = True
            candidates = [c for c in candidates if (c.get(field) or "") == value]
    if has_filter:
        return candidates
    return []


async def delete_check_item_backend(task_id, item_id, reason: str, auth: str) -> None:
    async with httpx.AsyncClient(base_url=TODO_BACKEND_BASE, timeout=10) as client:
        resp = await client.delete(
            f"/api/ai/ship-tasks/{task_id}/check-items/{item_id}",
            params={"source": "ai", "reason": reason},
            headers=backend_headers(auth),
        )
        resp.raise_for_status()


def local_delete_check_item(task_id, item_id) -> bool:
    items = LOCAL_CHECK_ITEMS.get(str(task_id), [])
    for idx, item in enumerate(items):
        if str(item.get("itemId")) == str(item_id):
            items.pop(idx)
            return True
    return False


async def stream_delete_check_item(
    req: ClassifyRequest, auth: str
) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    session_id = req.sessionId or int(datetime.now().timestamp() * 1000)
    if isinstance(session_id, str) and session_id.isdigit():
        session_id = int(session_id)
    turn_id = int(datetime.now().timestamp() * 1000) + 1
    user_message_id = turn_id + 1

    yield sse_event(
        "message_start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": None,
            "status": "running",
        },
    )

    task_id = req.taskId
    if isinstance(task_id, str) and task_id.isdigit():
        task_id = int(task_id)
    target = parse_delete_target(req.content or "", req.actionParams)

    def base_delta(status: str, content: str) -> dict:
        return {
            "seq": 1,
            "type": "action_result",
            "actionCode": "DELETE_CHECK_ITEM",
            "actionName": "删除检查项",
            "status": status,
            "content": content,
            "taskId": task_id,
            "deletedItem": None,
            "checkItemTotal": None,
            "check_list": [],
        }

    async def finish(delta: dict, extra_action_result: dict | None = None):
        yield sse_event("answer_delta", delta)
        if extra_action_result is not None:
            yield sse_event("action_result", extra_action_result)
        yield sse_event(
            "message_end",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "turnId": turn_id,
                "userMessageId": user_message_id,
                "assistantMessageId": user_message_id + 1,
                "status": "success",
                "actionCode": "DELETE_CHECK_ITEM",
            },
        )

    if task_id in (None, ""):
        async for chunk in finish(
            base_delta("rejected", "删除失败：该操作需要关联检验任务。")
        ):
            yield chunk
        return

    tree_source = "check_items_after_delete"
    use_local = False
    try:
        tree = await fetch_check_item_tree(task_id, auth)
    except Exception as e:
        print(
            f"[check-item] 后端检查项树接口不可达（{TODO_BACKEND_BASE}），降级为本地数据: {e}",
            flush=True,
        )
        use_local = True
        tree_source = "local_check_items"
        tree = local_check_item_tree(task_id)

    flat = flatten_check_item_tree(tree)
    total = count_check_items(tree)

    def delta_with_tree(status: str, content: str) -> dict:
        delta = base_delta(status, content)
        delta["checkItemTotal"] = total
        delta["check_list"] = tree
        delta["checkItemTree"] = tree
        return delta

    has_target = any(
        target.get(k)
        for k in (
            "itemId",
            "itemCode",
            "itemName",
            "targetText",
            "disciplineName",
            "sectionName",
            "riskLevel",
        )
    )
    if not has_target:
        async for chunk in finish(
            delta_with_tree("rejected", "删除失败：请提供要删除的检查项名称或编号。")
        ):
            yield chunk
        return

    matched = match_check_items(flat, target)
    if not matched:
        async for chunk in finish(
            delta_with_tree("rejected", "删除失败：未找到匹配的检查项。")
        ):
            yield chunk
        return
    if len(matched) > 1:
        delta = delta_with_tree(
            "rejected", "删除失败：匹配到多个检查项，请补充检查项编号或完整名称。"
        )
        delta["candidates"] = [
            {
                "itemId": i.get("itemId"),
                "itemCode": i.get("itemCode"),
                "itemName": i.get("itemName"),
            }
            for i in matched
        ]
        async for chunk in finish(delta):
            yield chunk
        return

    item = matched[0]
    reason = target.get("reason") or (req.content or "AI 根据用户要求删除检查项")[:200]
    if use_local:
        deleted = local_delete_check_item(task_id, item.get("itemId"))
    else:
        try:
            await delete_check_item_backend(task_id, item.get("itemId"), reason, auth)
            deleted = True
        except Exception as e:
            print(f"[check-item] 删除接口调用失败: {e}", flush=True)
            deleted = False
    if not deleted:
        async for chunk in finish(
            delta_with_tree("failed", "删除失败：删除检查项接口调用失败，请稍后重试。")
        ):
            yield chunk
        return

    if use_local:
        tree = local_check_item_tree(task_id)
    else:
        try:
            tree = await fetch_check_item_tree(task_id, auth)
        except Exception as e:
            print(f"[check-item] 删除后检查项树查询失败: {e}", flush=True)
            tree = []
    total = count_check_items(tree)

    deleted_item = {
        "itemId": item.get("itemId"),
        "itemCode": item.get("itemCode"),
        "itemName": item.get("itemName"),
        "disciplineName": item.get("disciplineName"),
        "sectionName": item.get("sectionName"),
        "riskLevel": item.get("riskLevel"),
        "riskLevelName": item.get("riskLevelName")
        or RISK_LEVEL_NAMES.get(item.get("riskLevel") or "", ""),
    }
    delta = base_delta("success", f"删除成功，现有 {total} 条检查项。")
    delta.update(
        {
            "deletedItem": deleted_item,
            "checkItemTotal": total,
            "check_list": tree,
            "checkItemTree": tree,
            "refresh": ["checkItems", "overview"],
            "source": tree_source,
        }
    )
    async for chunk in finish(
        delta,
        {
            "actionCode": "DELETE_CHECK_ITEM",
            "status": "success",
            "message": "删除成功",
            "payload": {
                "taskId": task_id,
                "deletedItem": deleted_item,
                "checkItemTotal": total,
            },
        },
    ):
        yield chunk


# ==================== 保存检查项（SAVE_CHECK_ITEMS） ====================

SAVE_CHECK_ITEMS_PATTERN = re.compile(
    r"(保存|落库).{0,30}(检[验查]项|清单)|(检[验查]项|清单).{0,20}(保存|落库)"
)


def detect_save_check_items(req: ClassifyRequest) -> bool:
    if req.actionCode == "SAVE_CHECK_ITEMS":
        return True
    if req.actionCode:
        return False
    text = req.content or ""
    if not SAVE_CHECK_ITEMS_PATTERN.search(text):
        return False
    if re.search(r"如何|怎么|怎样|能不能|可以.*吗|？|\?", text):
        return False
    return True


def normalize_check_list(raw) -> list:
    """把 check_list（扁平列表或检查项树）归一化为扁平检查项列表。"""
    flat = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        if "sections" in entry:
            for section in entry.get("sections") or []:
                for item in section.get("items") or []:
                    if isinstance(item, dict):
                        flat.append(
                            {
                                **item,
                                "disciplineCode": item.get("disciplineCode")
                                or entry.get("disciplineCode"),
                                "disciplineName": item.get("disciplineName")
                                or entry.get("disciplineName"),
                                "sectionName": item.get("sectionName")
                                or section.get("sectionName"),
                            }
                        )
        else:
            flat.append(entry)
    return flat


def build_save_item_body(item: dict) -> dict:
    body = {
        "itemCode": item.get("itemCode") or "",
        "itemName": item.get("itemName") or "",
        "riskLevel": item.get("riskLevel") or "low",
        "itemType": item.get("itemType") or "normal",
        "source": "ai",
    }
    for key in (
        "disciplineCode",
        "disciplineName",
        "sectionName",
        "categoryName",
        "regulationBasis",
        "actionRequirement",
    ):
        if item.get(key):
            body[key] = item[key]
    return body


async def stream_save_check_items(
    req: ClassifyRequest, auth: str
) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    session_id = req.sessionId or int(datetime.now().timestamp() * 1000)
    if isinstance(session_id, str) and session_id.isdigit():
        session_id = int(session_id)
    turn_id = int(datetime.now().timestamp() * 1000) + 1
    user_message_id = turn_id + 1

    yield sse_event(
        "message_start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": None,
            "status": "running",
        },
    )

    task_id = req.taskId
    if isinstance(task_id, str) and task_id.isdigit():
        task_id = int(task_id)
    params = req.actionParams or {}
    raw_list = params.get("check_list")
    if raw_list is None:
        raw_list = params.get("check_lsit")  # 兼容上游误传字段名

    def base_delta(status: str, content: str) -> dict:
        return {
            "seq": 1,
            "type": "action_result",
            "actionCode": "SAVE_CHECK_ITEMS",
            "actionName": "保存检查项",
            "status": status,
            "content": content,
            "taskId": task_id,
            "saveTotal": 0,
            "successTotal": 0,
            "failedTotal": 0,
            "savedItems": [],
            "failedItems": [],
        }

    async def finish(delta: dict, extra_action_result: dict | None = None):
        yield sse_event("answer_delta", delta)
        if extra_action_result is not None:
            yield sse_event("action_result", extra_action_result)
        yield sse_event(
            "message_end",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "turnId": turn_id,
                "userMessageId": user_message_id,
                "assistantMessageId": user_message_id + 1,
                "status": "success",
                "actionCode": "SAVE_CHECK_ITEMS",
            },
        )

    if task_id in (None, ""):
        delta = base_delta("rejected", "保存失败：该操作需要关联检验任务。")
        async for chunk in finish(delta):
            yield chunk
        return
    if raw_list is None:
        delta = base_delta("rejected", "保存失败：缺少 check_list，无法保存检查项。")
        async for chunk in finish(delta):
            yield chunk
        return

    items = normalize_check_list(raw_list)
    if not items:
        delta = base_delta(
            "rejected", "保存失败：check_list 为空，没有需要保存的检查项。"
        )
        async for chunk in finish(delta):
            yield chunk
        return

    saved_items = []
    failed_items = []
    backend_available = True
    for item in items:
        body = build_save_item_body(item)
        if not body["itemCode"]:
            failed_items.append(
                {
                    "itemCode": body["itemCode"],
                    "itemName": body["itemName"],
                    "reason": "itemCode 不能为空",
                }
            )
            continue
        if not body["itemName"]:
            failed_items.append(
                {
                    "itemCode": body["itemCode"],
                    "itemName": body["itemName"],
                    "reason": "itemName 不能为空",
                }
            )
            continue
        added = None
        if backend_available:
            try:
                added = await add_check_item_backend(task_id, body, auth)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                print(
                    f"[check-item] 后端新增接口不可达（{TODO_BACKEND_BASE}），降级为本地数据: {e}",
                    flush=True,
                )
                backend_available = False
            except Exception as e:
                failed_items.append(
                    {
                        "itemCode": body["itemCode"],
                        "itemName": body["itemName"],
                        "reason": f"新增接口调用失败: {e}",
                    }
                )
                continue
        if not backend_available:
            added = local_add_check_item(task_id, body)
        saved_items.append(
            {
                "itemId": (added or {}).get("itemId"),
                "itemCode": body["itemCode"],
                "itemName": body["itemName"],
            }
        )

    save_total = len(items)
    success_total = len(saved_items)
    failed_total = len(failed_items)
    if success_total == 0:
        status = "failed"
        content = f"保存失败：{save_total} 条检查项均未保存成功。"
    elif failed_total > 0:
        status = "partial_success"
        content = f"保存完成，成功新增 {success_total} 条，失败 {failed_total} 条。"
    else:
        status = "success"
        content = f"保存成功，已新增 {success_total} 条检查项。"

    delta = base_delta(status, content)
    delta.update(
        {
            "saveTotal": save_total,
            "successTotal": success_total,
            "failedTotal": failed_total,
            "savedItems": saved_items,
            "failedItems": failed_items,
            "refresh": ["checkItems", "overview"],
            "source": "check_items_after_save"
            if backend_available
            else "local_check_items",
        }
    )
    extra = None
    if success_total > 0:
        extra = {
            "actionCode": "SAVE_CHECK_ITEMS",
            "status": status,
            "message": content,
            "payload": {
                "taskId": task_id,
                "saveTotal": save_total,
                "successTotal": success_total,
                "failedTotal": failed_total,
                "savedItems": saved_items,
                "failedItems": failed_items,
            },
        }
    async for chunk in finish(delta, extra):
        yield chunk


# ==================== 生成检验单（GENERATE_PREPARATION_FORM） ====================

GENERATE_FORM_PATTERN = re.compile(
    r"生成.{0,20}(检验单|准备单|准备文档)|(检验单|准备单|准备文档).{0,10}生成|(检[验查]内容|检[验查]项).{0,10}生成一?份?(检验单|准备单)"
)
GENERATE_FORM_EXCLUDE_PATTERN = re.compile(
    r"如何|怎么|怎样|需要哪些|哪些步骤|是什么|什么是|能不能|可以.*吗|？|\?|查看|已经生成|推送"
)

LOCAL_PREPARATION_DOC_SEQ = {"value": 80000}


def detect_generate_preparation_form(req: ClassifyRequest) -> bool:
    if req.actionCode == "GENERATE_PREPARATION_FORM":
        return True
    if req.actionCode:
        return False
    text = req.content or ""
    if not GENERATE_FORM_PATTERN.search(text):
        return False
    if GENERATE_FORM_EXCLUDE_PATTERN.search(text):
        return False
    return True


async def generate_preparation_form_backend(task_id, auth: str) -> dict:
    async with httpx.AsyncClient(base_url=TODO_BACKEND_BASE, timeout=30) as client:
        resp = await client.post(
            f"/api/ai/ship-tasks/{task_id}/preparation/generate",
            headers=backend_headers(auth),
        )
        resp.raise_for_status()
        return (resp.json() or {}).get("data") or {}


def local_generate_preparation_form(task_id) -> dict:
    LOCAL_PREPARATION_DOC_SEQ["value"] += 1
    return {
        "docId": LOCAL_PREPARATION_DOC_SEQ["value"],
        "docNo": f"PREP-{task_id}-{datetime.now().strftime('%Y%m%d')}",
        "versionNo": 1,
        "status": "generated",
        "viewType": "surveyor",
    }


async def stream_generate_preparation_form(
    req: ClassifyRequest, auth: str
) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    session_id = req.sessionId or int(datetime.now().timestamp() * 1000)
    if isinstance(session_id, str) and session_id.isdigit():
        session_id = int(session_id)
    turn_id = int(datetime.now().timestamp() * 1000) + 1
    user_message_id = turn_id + 1

    yield sse_event(
        "message_start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": None,
            "status": "running",
        },
    )

    task_id = req.taskId
    if isinstance(task_id, str) and task_id.isdigit():
        task_id = int(task_id)

    def base_delta(status: str, content: str) -> dict:
        return {
            "seq": 1,
            "type": "action_result",
            "actionCode": "GENERATE_PREPARATION_FORM",
            "actionName": "生成检验单",
            "status": status,
            "content": content,
            "taskId": task_id,
            "preparationForm": None,
        }

    async def finish(delta: dict, extra_action_result: dict | None = None):
        yield sse_event("answer_delta", delta)
        if extra_action_result is not None:
            yield sse_event("action_result", extra_action_result)
        yield sse_event(
            "message_end",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "turnId": turn_id,
                "userMessageId": user_message_id,
                "assistantMessageId": user_message_id + 1,
                "status": "success",
                "actionCode": "GENERATE_PREPARATION_FORM",
            },
        )

    if task_id in (None, ""):
        async for chunk in finish(
            base_delta("rejected", "生成失败：该操作需要关联检验任务。")
        ):
            yield chunk
        return

    source = "generate_preparation_form"
    try:
        form = await generate_preparation_form_backend(task_id, auth)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        print(
            f"[preparation] 后端生成接口不可达（{TODO_BACKEND_BASE}），降级为本地数据: {e}",
            flush=True,
        )
        form = local_generate_preparation_form(task_id)
        source = "local_preparation_form"
    except Exception as e:
        print(f"[preparation] 检验单生成失败: {e}", flush=True)
        async for chunk in finish(
            base_delta("failed", "检验单生成失败，请稍后重试。")
        ):
            yield chunk
        return

    delta = base_delta("success", "检验单已生成，可进入预览页面查看。")
    delta.update(
        {
            "preparationForm": form or None,
            "refresh": ["preparationDocument", "overview"],
            "source": source,
        }
    )
    async for chunk in finish(
        delta,
        {
            "actionCode": "GENERATE_PREPARATION_FORM",
            "status": "success",
            "message": "检验单已生成",
            "payload": {"taskId": task_id, "preparationForm": form or None},
        },
    ):
        yield chunk


# ==================== 生成RA报告（GENERATE_RA_REPORT） ====================

GENERATE_RA_REPORT_PATTERN = re.compile(
    r"生成.{0,20}(RA\s*报告|检验报告|报告)|(RA\s*报告|检验报告|报告).{0,10}生成|(检[验查]内容|检[验查]项).{0,10}生成一?份?(RA\s*报告|检验报告|报告)",
    re.IGNORECASE,
)

LOCAL_RA_REPORT_DOC_SEQ = {"value": 81000}


def detect_generate_ra_report(req: ClassifyRequest) -> bool:
    if req.actionCode == "GENERATE_RA_REPORT":
        return True
    if req.actionCode:
        return False
    text = req.content or ""
    if not GENERATE_RA_REPORT_PATTERN.search(text):
        return False
    if GENERATE_FORM_EXCLUDE_PATTERN.search(text):
        return False
    return True


async def generate_ra_report_backend(task_id, auth: str) -> dict:
    async with httpx.AsyncClient(base_url=TODO_BACKEND_BASE, timeout=30) as client:
        resp = await client.post(
            f"/api/ai/ship-tasks/{task_id}/ra-report/generate",
            headers=backend_headers(auth),
        )
        resp.raise_for_status()
        return (resp.json() or {}).get("data") or {}


def local_generate_ra_report(task_id) -> dict:
    LOCAL_RA_REPORT_DOC_SEQ["value"] += 1
    return {
        "docId": LOCAL_RA_REPORT_DOC_SEQ["value"],
        "docNo": f"RA-{task_id}-{datetime.now().strftime('%Y%m%d')}",
        "versionNo": 1,
        "status": "generated",
        "viewType": "surveyor",
    }


async def stream_generate_ra_report(
    req: ClassifyRequest, auth: str
) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    session_id = req.sessionId or int(datetime.now().timestamp() * 1000)
    if isinstance(session_id, str) and session_id.isdigit():
        session_id = int(session_id)
    turn_id = int(datetime.now().timestamp() * 1000) + 1
    user_message_id = turn_id + 1

    yield sse_event(
        "message_start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": None,
            "status": "running",
        },
    )

    task_id = req.taskId
    if isinstance(task_id, str) and task_id.isdigit():
        task_id = int(task_id)

    def base_delta(status: str, content: str) -> dict:
        return {
            "seq": 1,
            "type": "action_result",
            "actionCode": "GENERATE_RA_REPORT",
            "actionName": "生成RA报告",
            "status": status,
            "content": content,
            "taskId": task_id,
            "raReport": None,
        }

    async def finish(delta: dict, extra_action_result: dict | None = None):
        yield sse_event("answer_delta", delta)
        if extra_action_result is not None:
            yield sse_event("action_result", extra_action_result)
        yield sse_event(
            "message_end",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "turnId": turn_id,
                "userMessageId": user_message_id,
                "assistantMessageId": user_message_id + 1,
                "status": "success",
                "actionCode": "GENERATE_RA_REPORT",
            },
        )

    if task_id in (None, ""):
        async for chunk in finish(
            base_delta("rejected", "生成失败：该操作需要关联检验任务。")
        ):
            yield chunk
        return

    source = "generate_ra_report"
    try:
        report = await generate_ra_report_backend(task_id, auth)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        print(
            f"[ra-report] 后端生成接口不可达（{TODO_BACKEND_BASE}），降级为本地数据: {e}",
            flush=True,
        )
        report = local_generate_ra_report(task_id)
        source = "local_ra_report"
    except Exception as e:
        print(f"[ra-report] RA报告生成失败: {e}", flush=True)
        async for chunk in finish(
            base_delta("failed", "RA报告生成失败，请稍后重试。")
        ):
            yield chunk
        return

    delta = base_delta("success", "RA报告已生成，可进入预览页面查看。")
    delta.update(
        {
            "raReport": report or None,
            "refresh": ["raReportDocument", "overview"],
            "source": source,
        }
    )
    async for chunk in finish(
        delta,
        {
            "actionCode": "GENERATE_RA_REPORT",
            "status": "success",
            "message": "RA报告已生成",
            "payload": {"taskId": task_id, "raReport": report or None},
        },
    ):
        yield chunk


# ==================== 上传资料（OPEN_UPLOAD_MATERIAL） ====================

UPLOAD_MATERIAL_PATTERN = re.compile(
    r"上传.{0,20}(资料|文件)|(资料|文件).{0,10}上传"
)
UPLOAD_MATERIAL_EXCLUDE_PATTERN = re.compile(
    r"如何|怎么|怎样|需要哪些|哪些步骤|是什么|什么是|能不能|可以.*吗|？|\?|查看|已经上传|下载"
)

LOCAL_MATERIAL_FILE_SEQ = {"value": 82000}


def detect_upload_material(req: ClassifyRequest) -> bool:
    if req.actionCode == "OPEN_UPLOAD_MATERIAL":
        return True
    if req.actionCode:
        return False
    text = req.content or ""
    if not UPLOAD_MATERIAL_PATTERN.search(text):
        return False
    if UPLOAD_MATERIAL_EXCLUDE_PATTERN.search(text):
        return False
    return True


async def upload_material_backend(task_id, auth: str) -> dict:
    async with httpx.AsyncClient(base_url=TODO_BACKEND_BASE, timeout=30) as client:
        resp = await client.post(
            f"/api/ai/ship-tasks/{task_id}/materials/upload",
            headers=backend_headers(auth),
        )
        resp.raise_for_status()
        return (resp.json() or {}).get("data") or {}


def local_upload_material(task_id) -> dict:
    LOCAL_MATERIAL_FILE_SEQ["value"] += 1
    return {
        "fileId": LOCAL_MATERIAL_FILE_SEQ["value"],
        "fileName": f"检验资料-{task_id}.pdf",
        "fileSize": 0,
        "status": "uploaded",
        "viewType": "surveyor",
    }


async def stream_upload_material(
    req: ClassifyRequest, auth: str
) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    session_id = req.sessionId or int(datetime.now().timestamp() * 1000)
    if isinstance(session_id, str) and session_id.isdigit():
        session_id = int(session_id)
    turn_id = int(datetime.now().timestamp() * 1000) + 1
    user_message_id = turn_id + 1

    yield sse_event(
        "message_start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": None,
            "status": "running",
        },
    )

    task_id = req.taskId
    if isinstance(task_id, str) and task_id.isdigit():
        task_id = int(task_id)

    def base_delta(status: str, content: str) -> dict:
        return {
            "seq": 1,
            "type": "action_result",
            "actionCode": "OPEN_UPLOAD_MATERIAL",
            "actionName": "上传资料",
            "status": status,
            "content": content,
            "taskId": task_id,
            "material": None,
        }

    async def finish(delta: dict, extra_action_result: dict | None = None):
        yield sse_event("answer_delta", delta)
        if extra_action_result is not None:
            yield sse_event("action_result", extra_action_result)
        yield sse_event(
            "message_end",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "turnId": turn_id,
                "userMessageId": user_message_id,
                "assistantMessageId": user_message_id + 1,
                "status": "success",
                "actionCode": "OPEN_UPLOAD_MATERIAL",
            },
        )

    if task_id in (None, ""):
        async for chunk in finish(
            base_delta("rejected", "操作失败：该操作需要关联检验任务。")
        ):
            yield chunk
        return

    source = "open_upload_material"
    try:
        material = await upload_material_backend(task_id, auth)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        print(
            f"[material] 后端上传接口不可达（{TODO_BACKEND_BASE}），降级为本地数据: {e}",
            flush=True,
        )
        material = local_upload_material(task_id)
        source = "local_upload_material"
    except Exception as e:
        print(f"[material] 资料上传失败: {e}", flush=True)
        async for chunk in finish(
            base_delta("failed", "资料上传失败，请稍后重试。")
        ):
            yield chunk
        return

    delta = base_delta("success", "资料已上传成功，可进入预览页面查看。")
    delta.update(
        {
            "material": material or None,
            "refresh": ["materialDocument", "overview"],
            "source": source,
        }
    )
    async for chunk in finish(
        delta,
        {
            "actionCode": "OPEN_UPLOAD_MATERIAL",
            "status": "success",
            "message": "资料已上传成功",
            "payload": {"taskId": task_id, "material": material or None},
        },
    ):
        yield chunk


# ==================== 生成工作日志（GENERATE_WORK_LOG） ====================

GENERATE_WORK_LOG_PATTERN = re.compile(
    r"生成.{0,20}(工作日志|检验日志|日志|工作记录)|(工作日志|检验日志|日志|工作记录).{0,10}生成|(检[验查]内容|检[验查]项).{0,10}生成一?份?(工作日志|检验日志|日志)"
)

LOCAL_WORK_LOG_DOC_SEQ = {"value": 83000}


def detect_generate_work_log(req: ClassifyRequest) -> bool:
    if req.actionCode == "GENERATE_WORK_LOG":
        return True
    if req.actionCode:
        return False
    text = req.content or ""
    if not GENERATE_WORK_LOG_PATTERN.search(text):
        return False
    if GENERATE_FORM_EXCLUDE_PATTERN.search(text):
        return False
    return True


async def generate_work_log_backend(task_id, auth: str) -> dict:
    async with httpx.AsyncClient(base_url=TODO_BACKEND_BASE, timeout=30) as client:
        resp = await client.post(
            f"/api/ai/ship-tasks/{task_id}/work-log/generate",
            headers=backend_headers(auth),
        )
        resp.raise_for_status()
        return (resp.json() or {}).get("data") or {}


def local_generate_work_log(task_id) -> dict:
    LOCAL_WORK_LOG_DOC_SEQ["value"] += 1
    return {
        "docId": LOCAL_WORK_LOG_DOC_SEQ["value"],
        "docNo": f"WL-{task_id}-{datetime.now().strftime('%Y%m%d')}",
        "versionNo": 1,
        "status": "generated",
        "viewType": "surveyor",
    }


async def stream_generate_work_log(
    req: ClassifyRequest, auth: str
) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    session_id = req.sessionId or int(datetime.now().timestamp() * 1000)
    if isinstance(session_id, str) and session_id.isdigit():
        session_id = int(session_id)
    turn_id = int(datetime.now().timestamp() * 1000) + 1
    user_message_id = turn_id + 1

    yield sse_event(
        "message_start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": None,
            "status": "running",
        },
    )

    task_id = req.taskId
    if isinstance(task_id, str) and task_id.isdigit():
        task_id = int(task_id)

    def base_delta(status: str, content: str) -> dict:
        return {
            "seq": 1,
            "type": "action_result",
            "actionCode": "GENERATE_WORK_LOG",
            "actionName": "生成工作日志",
            "status": status,
            "content": content,
            "taskId": task_id,
            "workLog": None,
        }

    async def finish(delta: dict, extra_action_result: dict | None = None):
        yield sse_event("answer_delta", delta)
        if extra_action_result is not None:
            yield sse_event("action_result", extra_action_result)
        yield sse_event(
            "message_end",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "turnId": turn_id,
                "userMessageId": user_message_id,
                "assistantMessageId": user_message_id + 1,
                "status": "success",
                "actionCode": "GENERATE_WORK_LOG",
            },
        )

    if task_id in (None, ""):
        async for chunk in finish(
            base_delta("rejected", "生成失败：该操作需要关联检验任务。")
        ):
            yield chunk
        return

    source = "generate_work_log"
    try:
        work_log = await generate_work_log_backend(task_id, auth)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        print(
            f"[work-log] 后端生成接口不可达（{TODO_BACKEND_BASE}），降级为本地数据: {e}",
            flush=True,
        )
        work_log = local_generate_work_log(task_id)
        source = "local_work_log"
    except Exception as e:
        print(f"[work-log] 工作日志生成失败: {e}", flush=True)
        async for chunk in finish(
            base_delta("failed", "工作日志生成失败，请稍后重试。")
        ):
            yield chunk
        return

    delta = base_delta("success", "工作日志已生成，可进入预览页面查看。")
    delta.update(
        {
            "workLog": work_log or None,
            "refresh": ["workLogDocument", "overview"],
            "source": source,
        }
    )
    async for chunk in finish(
        delta,
        {
            "actionCode": "GENERATE_WORK_LOG",
            "status": "success",
            "message": "工作日志已生成",
            "payload": {"taskId": task_id, "workLog": work_log or None},
        },
    ):
        yield chunk


# ==================== 开始检验（START_INSPECTION） ====================

START_INSPECTION_PATTERN = re.compile(
    r"(开始|进入|打开).{0,10}(检验|验船)|我要检验|开始验船|检验流程|检验页面"
)
START_INSPECTION_EXCLUDE_PATTERN = re.compile(
    r"如何|怎么|怎样|需要哪些|哪些步骤|是什么|什么是|能不能|可以.*吗|？|\?|查看|已经完成|已完成|推送|保存"
)


def detect_start_inspection(req: ClassifyRequest) -> bool:
    if req.actionCode == "START_INSPECTION":
        return True
    if req.actionCode:
        return False
    text = req.content or ""
    if not START_INSPECTION_PATTERN.search(text):
        return False
    if START_INSPECTION_EXCLUDE_PATTERN.search(text):
        return False
    return True


async def fetch_inspection_tasks(client_type: str, auth: str) -> list:
    async with httpx.AsyncClient(base_url=TODO_BACKEND_BASE, timeout=10) as client:
        resp = await client.get(
            "/api/ai/ship-tasks",
            params={
                "clientType": client_type,
                "status": "pending_inspection,inspection",
                "page": 1,
                "limit": 20,
            },
            headers=backend_headers(auth),
        )
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or {}
        return data.get("list") or []


def local_inspection_tasks() -> list:
    tasks = []
    task_id = 0
    for name, s in SHIPS.items():
        task_id += 1
        mapped = STATUS_CODE_MAP.get(s["状态"])
        if not mapped or mapped[0] not in ("pending_inspection", "inspection"):
            continue
        code, code_name = mapped
        tasks.append(
            {
                "taskId": task_id,
                "taskNo": f"LOCAL-TASK-{task_id:04d}",
                "shipName": name,
                "ccsNo": s.get("CCSNO", ""),
                "status": code,
                "statusName": code_name,
                "plannedInspectionDate": "",
                "surveyorName": "张工",
                "progressPercent": 0,
            }
        )
    return tasks


async def stream_start_inspection(
    req: ClassifyRequest, auth: str
) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    session_id = req.sessionId or int(datetime.now().timestamp() * 1000)
    if isinstance(session_id, str) and session_id.isdigit():
        session_id = int(session_id)
    turn_id = int(datetime.now().timestamp() * 1000) + 1
    user_message_id = turn_id + 1

    yield sse_event(
        "message_start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "userMessageId": user_message_id,
            "assistantMessageId": None,
            "status": "running",
        },
    )

    def base_delta(status: str, content: str) -> dict:
        return {
            "seq": 1,
            "type": "action_result",
            "actionCode": "START_INSPECTION",
            "actionName": "开始检验",
            "status": status,
            "content": content,
            "taskList": None,
        }

    async def finish(delta: dict, extra_action_result: dict | None = None):
        yield sse_event("answer_delta", delta)
        if extra_action_result is not None:
            yield sse_event("action_result", extra_action_result)
        yield sse_event(
            "message_end",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "turnId": turn_id,
                "userMessageId": user_message_id,
                "assistantMessageId": user_message_id + 1,
                "status": "success",
                "actionCode": "START_INSPECTION",
            },
        )

    source = "start_inspection"
    try:
        task_list = await fetch_inspection_tasks(req.clientType or "pc", auth)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        print(
            f"[inspection] 后端任务列表接口不可达（{TODO_BACKEND_BASE}），降级为本地数据: {e}",
            flush=True,
        )
        task_list = local_inspection_tasks()
        source = "local_ships"
    except Exception as e:
        print(f"[inspection] 任务列表查询失败: {e}", flush=True)
        async for chunk in finish(
            base_delta("failed", "获取任务列表失败，请稍后重试。")
        ):
            yield chunk
        return

    count = len(task_list)
    content = (
        f"已为您找到 {count} 项待检验和检验中的任务，请选择要检验的任务。"
        if count
        else "当前没有待检验或检验中的任务。"
    )
    delta = base_delta("success", content)
    delta.update(
        {
            "taskList": task_list,
            "refresh": ["inspectionTaskList", "overview"],
            "source": source,
        }
    )
    async for chunk in finish(
        delta,
        {
            "actionCode": "START_INSPECTION",
            "status": "success",
            "message": "查询成功",
            "payload": {"taskTotal": count, "taskList": task_list},
        },
    ):
        yield chunk


@app.post("/api/ai/chat/classify")
async def chat_classify(request: Request):
    raw = await request.body()
    if raw:
        try:
            data = json.loads(raw)
        except Exception as e:
            print(
                f"[classify] 请求体不是合法 JSON，按空请求处理: {e} body={raw.decode('utf-8', 'replace')[:2000]}",
                flush=True,
            )
            data = {}
    else:
        data = {}
    if not isinstance(data, dict):
        data = {}
    req = ClassifyRequest.model_validate(data)
    auth = request.headers.get("Authorization", "")
    if detect_start_inspection(req):
        stream = stream_start_inspection(req, auth)
    elif detect_generate_work_log(req):
        stream = stream_generate_work_log(req, auth)
    elif detect_upload_material(req):
        stream = stream_upload_material(req, auth)
    elif detect_generate_ra_report(req):
        stream = stream_generate_ra_report(req, auth)
    elif detect_generate_preparation_form(req):
        stream = stream_generate_preparation_form(req, auth)
    elif detect_save_check_items(req):
        stream = stream_save_check_items(req, auth)
    elif detect_delete_check_item(req):
        stream = stream_delete_check_item(req, auth)
    elif detect_add_check_item(req):
        stream = stream_add_check_item(req, auth)
    else:
        stream = stream_classify(req, auth)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ai/chat/stop")
async def chat_stop():
    return {"status": "stopped"}


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
    conv_id: str = ""


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
        first_user = re.sub(r"^【[^】]*】\s*", "", first_user) or "新对话"
        title = first_user.replace("\n", " ")[:24]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages_json = json.dumps(
        [{"role": m.role, "content": m.content} for m in req.messages],
        ensure_ascii=False,
    )
    convs = load_conversations()
    if req.conv_id:
        for c in convs:
            if c["id"] == req.conv_id:
                c["messages"] = messages_json
                c["time"] = now
                save_conversations(convs)
                return {"saved": True, "id": c["id"], "title": c["title"], "time": now}
    conv = {
        "id": req.conv_id or uuid.uuid4().hex,
        "title": title,
        "time": now,
        "messages": messages_json,
    }
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
