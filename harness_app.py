"""智能体 Harness 服务（LangGraph 辅助的观测→推荐→优化闭环）。

独立于主服务运行，不修改主服务任何代码：
  - 前端控制：提供 http://127.0.0.1:8144/assistant 静态页面，动态渲染指标、
    推荐动作和优化记录（页面内容随 harness 状态实时变化）。
  - 后端控制：所有对话经 harness 代理转发到主服务 /api/ai/chat/classify，
    转发前应用「动态覆盖规则」（意图关键词→actionCode），实现不改主服务
    代码的行为热调整。
  - Agent 本体自动优化：LangGraph StateGraph 编排
        observe（汇总交互与反馈指标）
     -> recommend（基于动作转移统计与业务流程图生成下一步推荐）
     -> optimize（依据负反馈自动生成/调整意图覆盖规则）
     -> apply（落盘 harness_state.json，前端立即生效）

运行:
  python -m uvicorn harness_app:app --port 8144
环境变量:
  HARNESS_MAIN_BASE  主服务地址，默认 http://127.0.0.1:8000
"""

import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, TypedDict

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.graph import END, START, StateGraph

from harness_llm_1 import annual_survey
from harness_llm_1 import harness as master_agent
from harness_llm_1.plugins import plugin_manager

plugin_manager.install_all()

MAIN_BASE = os.getenv("HARNESS_MAIN_BASE", "http://127.0.0.1:8000")
STATE_FILE = os.getenv("HARNESS_STATE_FILE", "harness_state.json")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 业务流程图：动作完成后推荐的后续动作（推荐引擎的先验）
WORKFLOW_NEXT: Dict[str, List[str]] = {
    "START_INSPECTION": ["ADD_CHECK_ITEM", "VIEW_CHECK_ITEMS_OVERVIEW", "GENERATE_PREPARATION_FORM"],
    "BEGIN_TASK": ["GENERATE_PREPARATION_FORM", "ADD_CHECK_ITEM", "OPEN_UPLOAD_MATERIAL"],
    "ADD_CHECK_ITEM": ["SAVE_CHECK_ITEMS", "VIEW_CHECK_ITEMS_OVERVIEW"],
    "DELETE_CHECK_ITEM": ["SAVE_CHECK_ITEMS", "VIEW_CHECK_ITEMS_OVERVIEW"],
    "SAVE_CHECK_ITEMS": ["VIEW_CHECK_ITEMS_OVERVIEW", "RECORD_ISSUE"],
    "VIEW_CHECK_ITEMS_OVERVIEW": ["RECORD_ISSUE", "GENERATE_RA_REPORT"],
    "GENERATE_PREPARATION_FORM": ["OPEN_UPLOAD_MATERIAL", "START_INSPECTION"],
    "GENERATE_RA_REPORT": ["GENERATE_WORK_LOG", "COMPLETE_INSPECTION_TASK"],
    "OPEN_UPLOAD_MATERIAL": ["GENERATE_RA_REPORT", "GENERATE_WORK_LOG"],
    "GENERATE_WORK_LOG": ["COMPLETE_INSPECTION_TASK"],
    "RECORD_ISSUE": ["CONFIRM_RECTIFICATION_ISSUES", "PENDING_RECTIFICATION_ISSUES", "VIEW_CHECK_ITEMS_OVERVIEW"],
    "CONFIRM_RECTIFICATION_ISSUES": ["COMPLETE_INSPECTION_TASK", "VIEW_CHECK_ITEMS_OVERVIEW"],
    "PENDING_RECTIFICATION_ISSUES": ["RECORD_ISSUE", "CONFIRM_RECTIFICATION_ISSUES"],
    "COMPLETE_INSPECTION_TASK": ["GENERATE_WORK_LOG"],
}

ACTION_NAMES = {
    "BEGIN_TASK": "船只任务准备",
    "START_INSPECTION": "开始检验",
    "ADD_CHECK_ITEM": "新增检查项",
    "DELETE_CHECK_ITEM": "删除检查项",
    "SAVE_CHECK_ITEMS": "保存检查项",
    "VIEW_CHECK_ITEMS_OVERVIEW": "查看检查项概览",
    "GENERATE_PREPARATION_FORM": "生成开检准备单",
    "GENERATE_RA_REPORT": "生成RA报告",
    "OPEN_UPLOAD_MATERIAL": "上传资料",
    "GENERATE_WORK_LOG": "生成工作日志",
    "RECORD_ISSUE": "问题记录",
    "CONFIRM_RECTIFICATION_ISSUES": "确认整改遗留问题",
    "PENDING_RECTIFICATION_ISSUES": "未整改遗留问题",
    "COMPLETE_INSPECTION_TASK": "完成检验任务",
}

ACTION_SAMPLE_CONTENT = {
    "BEGIN_TASK": "开始准备东海01船只任务",
    "START_INSPECTION": "开始检验",
    "ADD_CHECK_ITEM": "新增检查项：编号A-101，名称救生设备检查",
    "DELETE_CHECK_ITEM": "删除检查项A-101",
    "SAVE_CHECK_ITEMS": "保存检查项",
    "VIEW_CHECK_ITEMS_OVERVIEW": "查看检查项概览",
    "GENERATE_PREPARATION_FORM": "生成开检准备单",
    "GENERATE_RA_REPORT": "生成RA报告",
    "OPEN_UPLOAD_MATERIAL": "上传资料",
    "GENERATE_WORK_LOG": "生成工作日志",
    "RECORD_ISSUE": "发现船体外板存在裂纹，严重程度高",
    "CONFIRM_RECTIFICATION_ISSUES": "确认这些遗留问题已经整改",
    "PENDING_RECTIFICATION_ISSUES": "这些遗留问题标记为未整改",
    "COMPLETE_INSPECTION_TASK": "帮我完成检验任务",
}


# ==================== Harness 持久状态 ====================
def default_state() -> Dict[str, Any]:
    return {
        "interactions": [],       # 交互记录
        "feedback": [],           # 用户反馈
        "override_rules": [],     # 动态意图覆盖规则 [{keyword, actionCode, source, createdAt}]
        "metrics": {},            # observe 输出
        "recommendations": [],    # recommend 输出
        "optimizations": [],      # optimize 历史
        "last_optimized_at": None,
        "pipeline_runs": [],     # 管线运行历史
        "rec_stats": {"shown": 0, "adopted": 0, "bySource": {}},  # 推荐展示/采纳统计
    }


def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            base = default_state()
            base.update(data if isinstance(data, dict) else {})
            return base
        except (json.JSONDecodeError, OSError):
            pass
    return default_state()


def save_state() -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(HARNESS_STATE, f, ensure_ascii=False, indent=2)


HARNESS_STATE = load_state()


# ==================== LangGraph 优化管线 ====================
class HarnessGraphState(TypedDict, total=False):
    interactions: List[Dict[str, Any]]
    feedback: List[Dict[str, Any]]
    override_rules: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    new_rules: List[Dict[str, Any]]
    notes: List[str]


def node_observe(state: HarnessGraphState) -> HarnessGraphState:
    """观测：汇总交互与反馈，形成指标"""
    interactions = state.get("interactions", [])
    feedback = state.get("feedback", [])
    total = len(interactions)
    action_counts: Dict[str, int] = {}
    success = 0
    for it in interactions:
        code = it.get("actionCode") or "UNMATCHED"
        action_counts[code] = action_counts.get(code, 0) + 1
        if it.get("status") == "success":
            success += 1
    negative = [fb for fb in feedback if fb.get("rating") == "down"]
    metrics = {
        "totalInteractions": total,
        "successRate": round(success / total, 3) if total else None,
        "actionCounts": action_counts,
        "negativeFeedback": len(negative),
        "unmatchedCount": action_counts.get("UNMATCHED", 0),
        "observedAt": datetime.now().isoformat(),
    }
    notes = state.get("notes", []) + [f"observe: {total}次交互, {len(negative)}条负反馈"]
    return {"metrics": metrics, "notes": notes}


def node_recommend(state: HarnessGraphState) -> HarnessGraphState:
    """推荐：业务流程先验 + 实际动作转移统计，给出下一步动作推荐"""
    interactions = state.get("interactions", [])
    transitions: Dict[str, Dict[str, int]] = {}
    prev = None
    for it in interactions:
        code = it.get("actionCode")
        if prev and code and prev != code:
            transitions.setdefault(prev, {})
            transitions[prev][code] = transitions[prev].get(code, 0) + 1
        if code:
            prev = code
    last_action = next(
        (it.get("actionCode") for it in reversed(interactions) if it.get("actionCode")),
        None,
    )
    candidates: List[str] = []
    if last_action:
        learned = sorted(
            transitions.get(last_action, {}).items(), key=lambda x: -x[1])
        candidates += [a for a, _ in learned]
        candidates += WORKFLOW_NEXT.get(last_action, [])
    else:
        candidates += ["START_INSPECTION", "BEGIN_TASK", "VIEW_CHECK_ITEMS_OVERVIEW"]
    seen, recs = set(), []
    for code in candidates:
        if code in seen or code not in ACTION_NAMES:
            continue
        seen.add(code)
        recs.append({
            "actionCode": code,
            "actionName": ACTION_NAMES[code],
            "sampleContent": ACTION_SAMPLE_CONTENT.get(code, ""),
            "reason": ("基于历史动作转移统计" if code in transitions.get(last_action or "", {})
                       else "基于船检业务流程"),
        })
        if len(recs) >= 4:
            break
    notes = state.get("notes", []) + [f"recommend: 基于last_action={last_action} 生成{len(recs)}条推荐"]
    return {"recommendations": recs, "notes": notes}


def _extract_keyword(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[，。！？,.!?\s]+$", "", text)
    return text[:24]


def node_optimize(state: HarnessGraphState) -> HarnessGraphState:
    """优化：依据负反馈自动生成意图覆盖规则（agent 行为自动修正）"""
    feedback = state.get("feedback", [])
    rules = list(state.get("override_rules", []))
    existing = {(r["keyword"], r["actionCode"]) for r in rules}
    new_rules: List[Dict[str, Any]] = []
    for fb in feedback:
        if fb.get("rating") != "down" or fb.get("applied"):
            continue
        expected = fb.get("expectedAction")
        keyword = _extract_keyword(fb.get("content", ""))
        if not expected or not keyword or expected not in ACTION_NAMES:
            continue
        if (keyword, expected) in existing:
            fb["applied"] = True
            continue
        rule = {
            "keyword": keyword,
            "actionCode": expected,
            "source": "auto_optimize",
            "fromFeedback": fb.get("id"),
            "createdAt": datetime.now().isoformat(),
        }
        rules.append(rule)
        new_rules.append(rule)
        existing.add((keyword, expected))
        fb["applied"] = True
    notes = state.get("notes", []) + [f"optimize: 新增{len(new_rules)}条覆盖规则"]
    return {"override_rules": rules, "new_rules": new_rules,
            "feedback": feedback, "notes": notes}


def node_apply(state: HarnessGraphState) -> HarnessGraphState:
    """应用：把优化结果写回 harness 持久状态"""
    HARNESS_STATE["metrics"] = state.get("metrics", {})
    HARNESS_STATE["recommendations"] = state.get("recommendations", [])
    HARNESS_STATE["override_rules"] = state.get("override_rules", [])
    HARNESS_STATE["feedback"] = state.get("feedback", HARNESS_STATE["feedback"])
    if state.get("new_rules"):
        HARNESS_STATE["optimizations"].append({
            "at": datetime.now().isoformat(),
            "newRules": state["new_rules"],
            "notes": state.get("notes", []),
        })
    HARNESS_STATE["last_optimized_at"] = datetime.now().isoformat()
    save_state()
    return {"notes": state.get("notes", []) + ["apply: 状态已落盘"]}


def build_graph():
    g = StateGraph(HarnessGraphState)
    g.add_node("observe", node_observe)
    g.add_node("recommend", node_recommend)
    g.add_node("optimize", node_optimize)
    g.add_node("apply", node_apply)
    g.add_edge(START, "observe")
    g.add_edge("observe", "recommend")
    g.add_edge("recommend", "optimize")
    g.add_edge("optimize", "apply")
    g.add_edge("apply", END)
    return g.compile()


HARNESS_GRAPH = build_graph()


def run_pipeline() -> Dict[str, Any]:
    result = HARNESS_GRAPH.invoke({
        "interactions": HARNESS_STATE["interactions"],
        "feedback": HARNESS_STATE["feedback"],
        "override_rules": HARNESS_STATE["override_rules"],
        "notes": [],
    })
    recs = result.get("recommendations", [])
    stats = HARNESS_STATE.setdefault("rec_stats", {"shown": 0, "adopted": 0, "bySource": {}})
    stats["shown"] += len(recs)
    for r in recs:
        src = r.get("reason", "其他")
        stats["bySource"][src] = stats["bySource"].get(src, 0) + 1
    HARNESS_STATE.setdefault("pipeline_runs", []).append({
        "at": datetime.now().isoformat(),
        "notes": result.get("notes", []),
        "newRules": len(result.get("new_rules", [])),
        "recommendations": len(recs),
    })
    HARNESS_STATE["pipeline_runs"] = HARNESS_STATE["pipeline_runs"][-50:]
    save_state()
    return {
        "metrics": result.get("metrics", {}),
        "recommendations": result.get("recommendations", []),
        "newRules": result.get("new_rules", []),
        "overrideRules": HARNESS_STATE["override_rules"],
        "pipelineNotes": result.get("notes", []),
    }


# ==================== FastAPI 应用 ====================
app = FastAPI(title="Agent Harness", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
@app.get("/index.html")
async def index_page():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/ship-inspection")
@app.get("/ship-inspection.html")
async def ship_inspection_page():
    return FileResponse(os.path.join(STATIC_DIR, "ship-inspection.html"))


@app.get("/assistant")
async def assistant_page():
    return FileResponse(os.path.join(STATIC_DIR, "assistant.html"))


@app.get("/annual-survey")
@app.get("/annual-survey.html")
async def annual_survey_page():
    return FileResponse(os.path.join(STATIC_DIR, "annual-survey.html"))


# ==================== 年度检验（ship-annual-survey Skill） ====================
@app.get("/api/survey/tasks")
async def survey_tasks():
    return JSONResponse({"tasks": annual_survey.list_tasks(),
                         "stages": annual_survey.STAGES,
                         "stageNames": annual_survey.STAGE_NAMES,
                         "confirmPoints": annual_survey.CONFIRM_POINTS})


@app.post("/api/survey/tasks")
async def survey_create(request: Request):
    data = await request.json()
    if not (data.get("ship_identifier") or "").strip():
        return JSONResponse({"error": "请提供船名、IMO 编号或检验工作号"},
                            status_code=400)
    task = annual_survey.create_task(data)
    return JSONResponse({"task": task})


@app.get("/api/survey/tasks/{task_id}")
async def survey_get(task_id: str):
    task = annual_survey.load_task(task_id)
    if task is None:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    return JSONResponse({"task": task})


@app.post("/api/survey/tasks/{task_id}/advance")
async def survey_advance(task_id: str):
    result = await annual_survey.advance_task(task_id)
    status = 404 if result.get("error") else 200
    return JSONResponse(result, status_code=status)


@app.post("/api/survey/tasks/{task_id}/confirm")
async def survey_confirm(task_id: str, request: Request):
    data = await request.json()
    result = annual_survey.confirm_point(
        task_id, data.get("point", ""), bool(data.get("confirmed", True)))
    status = 404 if result.get("error") else 200
    return JSONResponse(result, status_code=status)


@app.post("/api/survey/tasks/{task_id}/records")
async def survey_record(task_id: str, request: Request):
    data = await request.json()
    result = annual_survey.add_onboard_record(task_id, data)
    status = 404 if result.get("error") else 200
    return JSONResponse(result, status_code=status)


@app.get("/api/harness/state")
async def harness_state():
    return JSONResponse({
        "mainBase": MAIN_BASE,
        "metrics": HARNESS_STATE["metrics"],
        "recommendations": HARNESS_STATE["recommendations"],
        "overrideRules": HARNESS_STATE["override_rules"],
        "corrections": master_agent.memory.corrections(),
        "optimizations": HARNESS_STATE["optimizations"][-10:],
        "lastOptimizedAt": HARNESS_STATE["last_optimized_at"],
        "recentInteractions": HARNESS_STATE["interactions"][-20:],
    })


@app.get("/api/harness/analytics")
async def harness_analytics():
    """数据分析大盘：时序趋势、动作分布、推荐分析、管线运行历史"""
    interactions = HARNESS_STATE["interactions"]
    buckets: Dict[str, Dict[str, int]] = {}
    for it in interactions:
        key = (it.get("at") or "")[:16]  # 分钟级
        b = buckets.setdefault(key, {"total": 0, "success": 0, "override": 0})
        b["total"] += 1
        if it.get("status") == "success":
            b["success"] += 1
        if it.get("overrideRule"):
            b["override"] += 1
    timeline = [
        {"time": k[11:], "total": v["total"], "success": v["success"], "override": v["override"]}
        for k, v in sorted(buckets.items())
    ][-30:]
    action_dist: Dict[str, int] = {}
    status_dist: Dict[str, int] = {}
    for it in interactions:
        action_dist[it.get("actionCode") or "UNMATCHED"] = action_dist.get(it.get("actionCode") or "UNMATCHED", 0) + 1
        status_dist[it.get("status") or "unknown"] = status_dist.get(it.get("status") or "unknown", 0) + 1
    stats = HARNESS_STATE.get("rec_stats", {})
    shown = stats.get("shown", 0)
    return JSONResponse({
        "timeline": timeline,
        "actionDistribution": sorted(action_dist.items(), key=lambda x: -x[1]),
        "statusDistribution": status_dist,
        "recStats": {
            "shown": shown,
            "adopted": stats.get("adopted", 0),
            "adoptionRate": round(stats.get("adopted", 0) / shown, 3) if shown else None,
            "bySource": stats.get("bySource", {}),
        },
        "pipelineRuns": HARNESS_STATE.get("pipeline_runs", [])[-10:],
        "ruleCount": len(HARNESS_STATE["override_rules"]) + len(
            master_agent.memory.corrections()),
        "feedbackCount": len(HARNESS_STATE["feedback"]),
        "aiAnalysis": await master_agent.analyze_dashboard({
            "actionDistribution": action_dist,
            "statusDistribution": status_dist,
            "recStats": stats,
            "totalInteractions": len(interactions),
            "corrections": len(master_agent.memory.corrections()),
        }),
    })


@app.post("/api/harness/optimize")
async def harness_optimize():
    result = run_pipeline()
    # 推荐由主 agent（推荐子 agent）接管，规则推荐仅在 LLM 不可用时兜底
    inter = HARNESS_STATE["interactions"]
    last_action = next((i["actionCode"] for i in reversed(inter)
                        if i.get("actionCode")), None)
    llm_recs = await master_agent.recommend_next(last_action, inter)
    if llm_recs:
        for r in llm_recs:
            r["reason"] = ("主agent推荐：" + r["reason"]
                           if r.get("source") == "llm" else r["reason"])
        HARNESS_STATE["recommendations"] = llm_recs
        result["recommendations"] = llm_recs
        save_state()
    return JSONResponse(result)


@app.post("/api/harness/feedback")
async def harness_feedback(request: Request):
    data = await request.json()
    fb = {
        "id": str(uuid.uuid4()),
        "interactionId": data.get("interactionId"),
        "content": data.get("content", ""),
        "actionCode": data.get("actionCode"),
        "expectedAction": data.get("expectedAction"),
        "rating": data.get("rating", "down"),
        "applied": False,
        "at": datetime.now().isoformat(),
    }
    HARNESS_STATE["feedback"].append(fb)
    save_state()
    # 负反馈 → 写入主 agent 跨会话纠错记忆（下一轮 System Prompt 即生效）
    if fb["rating"] == "down" and fb.get("expectedAction"):
        master_agent.learn_from_feedback(fb["content"], fb["expectedAction"])
    # 负反馈立即触发一轮自动优化
    result = run_pipeline() if fb["rating"] == "down" else None
    return JSONResponse({"feedback": fb, "optimizeResult": result})


def apply_override_rules(payload: Dict[str, Any]) -> Dict[str, Any]:
    """转发前应用动态覆盖规则：命中关键词则显式指定 actionCode"""
    if payload.get("actionCode"):
        return payload
    content = payload.get("content", "") or ""
    for rule in HARNESS_STATE["override_rules"]:
        if rule["keyword"] and rule["keyword"] in content:
            payload = dict(payload)
            payload["actionCode"] = rule["actionCode"]
            payload["_harnessRule"] = rule["keyword"]
            break
    return payload


@app.post("/api/harness/chat")
async def harness_chat(request: Request):
    raw = await request.body()
    try:
        payload = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    applied = apply_override_rules(payload)
    rule_hit = applied.pop("_harnessRule", None)
    auth = request.headers.get("Authorization", "")
    content_in = payload.get("content", "") or ""
    # 主 agent（DeepSeek）决策动作；覆盖规则/显式 actionCode 作为辅助快速通道
    decision = None
    if not applied.get("actionCode") and content_in:
        inter = HARNESS_STATE["interactions"]
        last_action = next((i["actionCode"] for i in reversed(inter)
                            if i.get("actionCode")), None)
        decision = await master_agent.decide_action(content_in, {
            "pageCode": payload.get("pageCode"),
            "sessionType": payload.get("sessionType"),
            "lastAction": last_action,
        })
        if decision["actionCode"] != "GENERAL_QA":
            applied = dict(applied)
            applied["actionCode"] = decision["actionCode"]
            if decision.get("shipName"):
                params = dict(applied.get("actionParams") or {})
                params.setdefault("shipName", decision["shipName"])
                applied["actionParams"] = params
    # 年度检验流程为 harness 本地 Skill，不转发主服务
    if decision and decision["actionCode"] == "ANNUAL_SURVEY":
        ship = decision.get("shipName") or ""
        task = annual_survey.create_task({"ship_identifier": ship}) if ship else None
        answer = (
            f"已为「{ship}」创建年度检验任务（工作号 "
            f"{task['input']['inspection_work_order']}），"
            f"请打开 /annual-survey?taskId={task['id']} 推进流程。"
            if task else
            "要发起年度检验，请提供船名、IMO 编号或检验工作号；"
            "也可以直接打开 /annual-survey 页面创建任务。")
        HARNESS_STATE["interactions"].append({
            "id": str(uuid.uuid4()),
            "at": datetime.now().isoformat(),
            "content": content_in,
            "requestActionCode": None,
            "overrideRule": None,
            "decidedBy": decision.get("decidedBy"),
            "decideReason": decision.get("reason"),
            "confidence": decision.get("confidence"),
            "actionCode": "ANNUAL_SURVEY",
            "status": "success",
        })
        save_state()

        async def local_reply() -> AsyncGenerator[bytes, None]:
            data = json.dumps({"content": answer, "actionCode": "ANNUAL_SURVEY",
                               "status": "success"}, ensure_ascii=False)
            yield f"event: answer_delta\ndata: {data}\n\n".encode()
            meta = json.dumps({
                "actionCode": "ANNUAL_SURVEY",
                "surveyTaskId": task["id"] if task else None,
                "decidedBy": decision.get("decidedBy"),
                "decideReason": decision.get("reason"),
                "confidence": decision.get("confidence"),
            }, ensure_ascii=False)
            yield f"event: harness_meta\ndata: {meta}\n\n".encode()

        return StreamingResponse(local_reply(), media_type="text/event-stream")
    if payload.get("fromRecommendation") or any(
        r.get("sampleContent") == content_in
        for r in HARNESS_STATE.get("recommendations", [])
    ):
        stats = HARNESS_STATE.setdefault("rec_stats", {"shown": 0, "adopted": 0, "bySource": {}})
        stats["adopted"] += 1
    interaction = {
        "id": str(uuid.uuid4()),
        "at": datetime.now().isoformat(),
        "content": payload.get("content", ""),
        "requestActionCode": payload.get("actionCode") or None,
        "overrideRule": rule_hit,
        "decidedBy": (decision or {}).get("decidedBy"),
        "decideReason": (decision or {}).get("reason"),
        "confidence": (decision or {}).get("confidence"),
        "actionCode": None,
        "status": None,
    }

    async def relay() -> AsyncGenerator[bytes, None]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if auth:
            headers["Authorization"] = auth
        try:
            async with httpx.AsyncClient(base_url=MAIN_BASE, timeout=180) as client:
                async with client.stream(
                    "POST", "/api/ai/chat/classify", json=applied, headers=headers
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        text = chunk.decode("utf-8", errors="ignore")
                        if interaction["actionCode"] is None:
                            m = re.search(r'"actionCode"\s*:\s*"([A-Z_]+)"', text)
                            if m:
                                interaction["actionCode"] = m.group(1)
                        if interaction["status"] is None:
                            m = re.search(r'"status"\s*:\s*"(success|rejected|failed)"', text)
                            if m:
                                interaction["status"] = m.group(1)
                        yield chunk
        except httpx.HTTPError as e:
            err = json.dumps(
                {"content": f"[harness] 主服务不可达（{MAIN_BASE}）: {e}"},
                ensure_ascii=False)
            interaction["status"] = "failed"
            yield f"event: answer_delta\ndata: {err}\n\n".encode()
        finally:
            meta = json.dumps({
                "interactionId": interaction["id"],
                "actionCode": interaction["actionCode"],
                "overrideRule": rule_hit,
                "decidedBy": interaction["decidedBy"],
                "decideReason": interaction["decideReason"],
                "confidence": interaction["confidence"],
            }, ensure_ascii=False)
            yield f"event: harness_meta\ndata: {meta}\n\n".encode()
            HARNESS_STATE["interactions"].append(interaction)
            HARNESS_STATE["interactions"] = HARNESS_STATE["interactions"][-500:]
            save_state()

    return StreamingResponse(relay(), media_type="text/event-stream")


# 其余 /api/* 一律代理到主服务（/api/chat、/api/ships、/api/conversations 等），
# 使 harness 单端口即可访问全部页面功能
@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_main(rest: str, request: Request):
    url = f"/api/{rest}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() in ("authorization", "content-type", "accept")}

    async def relay_stream() -> AsyncGenerator[bytes, None]:
        try:
            async with httpx.AsyncClient(base_url=MAIN_BASE, timeout=300) as client:
                async with client.stream(
                    request.method, url, params=request.query_params,
                    content=body or None, headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except httpx.HTTPError as e:
            err = json.dumps(
                {"error": f"[harness] 主服务不可达（{MAIN_BASE}）: {e}，"
                          "请先启动主服务（python main_multi_agent.py）"},
                ensure_ascii=False)
            yield f"data: {err}\n\ndata: [DONE]\n\n".encode()

    if "text/event-stream" in request.headers.get("accept", "") or rest == "chat":
        return StreamingResponse(relay_stream(), media_type="text/event-stream")
    try:
        async with httpx.AsyncClient(base_url=MAIN_BASE, timeout=60) as client:
            resp = await client.request(
                request.method, url, params=request.query_params,
                content=body or None, headers=headers)
        return JSONResponse(
            resp.json() if resp.headers.get("content-type", "").startswith(
                "application/json") else {"raw": resp.text},
            status_code=resp.status_code)
    except httpx.HTTPError as e:
        return JSONResponse(
            {"error": f"[harness] 主服务不可达（{MAIN_BASE}）: {e}，"
                      "请先启动主服务（python main_multi_agent.py）"},
            status_code=502)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8144)
