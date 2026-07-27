"""ship-annual-survey Skill：营运船舶年度检验全流程编排。

LangGraph 状态机编排六个阶段：
  profile（船舶档案调取）→ checklist（检查项生成）→ preparation（检验前准备）
  → onboard（登轮检验）→ records（检验记录接收）→ certificate（证书报告编制与归档）

流程负责编排与成果生成，不代替验船师做缺陷定性、证书签发与归档决策：
每个关键节点设人工确认点，未确认不得进入下一阶段；
证书校验不通过则阻断自动归档；数据缺失输出缺失清单、不补造。
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from . import config
from .hooks import hooks
from .subagent import SubAgent, parse_json_loose
from .tools import registry
from . import survey_tools  # noqa: F401  注册 requires_tools

SURVEYS_DIR = os.path.join(config.OUTPUT_DIR, "surveys")
os.makedirs(SURVEYS_DIR, exist_ok=True)

STAGES = ["profile", "checklist", "preparation", "onboard", "records",
          "certificate"]

STAGE_NAMES = {
    "profile": "船舶档案调取",
    "checklist": "检查项生成",
    "preparation": "检验前准备",
    "onboard": "登轮检验",
    "records": "检验记录接收",
    "certificate": "证书报告编制与归档",
}

# 人工确认点（验船师确认后流程才能继续）
CONFIRM_POINTS = {
    "priority_checklist": "重点检查项确认",
    "preparation_reports": "两类检前报告确认",
    "owner_notice_send": "船东通知发送确认",
    "issue_qualification": "现场问题定性确认",
    "equipment_changes": "新增设备信息确认",
    "certificate_opinion": "证书处理意见确认",
    "final_archive": "最终报告及归档确认",
}

# 各阶段进入下一步前必须完成的确认点
STAGE_GATE: Dict[str, List[str]] = {
    "checklist": ["priority_checklist"],
    "preparation": ["preparation_reports", "owner_notice_send"],
    "onboard": ["issue_qualification"],
    "records": ["equipment_changes"],
    "certificate": ["certificate_opinion", "final_archive"],
}

REPORT_WRITER = SubAgent(
    "survey-report-writer",
    "你是中国船级社验船师助理。基于给定的年度检验数据 JSON，"
    "生成简洁的中文报告要点。输出 JSON："
    '{"surveyorReport": "验船师登轮准备报告要点（换行分条）", '
    '"ownerNotice": "船东准备通知要点（换行分条）"}，不要输出其他内容。',
)


def _tool(name: str, retries: int = 1, **kwargs: Any) -> Any:
    """接口调用：失败重试一次，仍失败标记转人工"""
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            return registry.get(name).run(**kwargs)
        except Exception as e:  # 接口失败按异常处理规范兜底
            last_err = e
    hooks.fire("on_error", {"error": f"工具 {name} 调用失败: {last_err}"})
    return {"error": str(last_err), "manualHandover": True}


class SurveyState(TypedDict, total=False):
    task: Dict[str, Any]


# ---------------- 六阶段节点 ----------------
def node_profile(state: SurveyState) -> SurveyState:
    task = state["task"]
    ship = task["input"]["ship_identifier"]
    profile = _tool("get_ship_profile", ship_identifier=ship)
    certs = _tool("get_certificate_status", ship_identifier=ship)
    memos = _tool("get_outstanding_memos", ship_identifier=ship)
    missing: List[str] = []
    if not profile.get("found"):
        missing.append("船舶档案")
    if not certs.get("found"):
        missing.append("证书记录")
    task["outputs"]["ship_profile_summary"] = {
        "profile": profile,
        "certificates": certs,
        "outstandingMemos": memos,
        "missing": missing,  # 数据缺失：输出缺失清单，不补造
    }
    task["stage_status"]["profile"] = "done" if not missing else "attention"
    return {"task": task}


def node_checklist(state: SurveyState) -> SurveyState:
    task = state["task"]
    ship = task["input"]["ship_identifier"]
    base = _tool("generate_base_checklist", ship_identifier=ship)
    memos = _tool("get_outstanding_memos", ship_identifier=ship)
    risks = _tool("get_ship_risk_history", ship_identifier=ship)
    reg_query = " ".join(
        [i.get("name", "") for i in base if isinstance(i, dict)]
        + [m.get("issue", "") for m in memos if isinstance(m, dict)])
    regs = _tool("search_maritime_regulations", query=reg_query)
    priority = []
    for m in memos:
        if isinstance(m, dict) and m.get("needRecheck"):
            priority.append({
                "itemNo": m["memoNo"], "name": f"遗留复核：{m['issue']}",
                "category": "遗留备忘", "risk": "高风险",
                "source": "outstanding_memo", "status": "pending",
            })
    for i in base:
        if isinstance(i, dict) and i.get("risk") == "高风险":
            priority.append(dict(i, source="risk_model"))
    task["outputs"]["base_checklist"] = base
    task["outputs"]["priority_checklist"] = {
        "items": priority,
        "riskHistory": risks,
        "regulationUpdates": regs,
        "fieldSuggestions": [
            f"重点核查：{p['name']}（依据 {regs[0]['ref'] if regs else 'CCS规范'}）"
            for p in priority[:5]
        ],
    }
    task["stage_status"]["checklist"] = "awaiting_confirm"
    return {"task": task}


async def _llm_reports(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        data = parse_json_loose(await REPORT_WRITER.run(
            json.dumps(payload, ensure_ascii=False)))
        if isinstance(data, dict) and data.get("surveyorReport"):
            return data
    except (httpx.HTTPError, ValueError, KeyError) as e:
        hooks.fire("on_error", {"error": f"报告子agent失败: {e}"})
    return None


async def node_preparation(state: SurveyState) -> SurveyState:
    task = state["task"]
    summary = task["outputs"].get("ship_profile_summary", {})
    priority = task["outputs"].get("priority_checklist", {})
    llm = await _llm_reports({
        "shipProfile": summary.get("profile"),
        "certificates": summary.get("certificates"),
        "priorityItems": priority.get("items"),
        "regulations": priority.get("regulationUpdates"),
    })
    profile = summary.get("profile", {})
    regs = priority.get("regulationUpdates", [])
    surveyor_report = (llm or {}).get("surveyorReport") or "\n".join(
        ["【验船师登轮准备报告】",
         f"船舶：{profile.get('shipName')}（{profile.get('shipType')}，"
         f"建造 {profile.get('builtDate')}）",
         f"证书：{'均有效' if summary.get('certificates', {}).get('issuable') else '存在异常，需复核'}",
         f"历史问题：{len(summary.get('outstandingMemos', []))} 项遗留备忘待复核",
         f"重点检查项：{len(priority.get('items', []))} 项（含遗留复核与高风险项）",
         "法规依据：" + "；".join(f"{r['ref']} {r['update']}" for r in regs[:3]),
         "现场建议：优先复核遗留缺陷，核对证书原件与维护记录"])
    owner_notice = (llm or {}).get("ownerNotice") or "\n".join(
        ["【船东准备通知】",
         "1. 备齐全套证书文件原件及维护保养记录",
         "2. 提供遗留问题整改材料与佐证照片",
         "3. 安排轮机、甲板部人员配合设备现场试验",
         f"4. 计划登轮时间：{task['input'].get('planned_boarding_time', '待定')}，"
         "请提前开放相关舱室并做好安全防护"])
    task["outputs"]["surveyor_preparation_report"] = surveyor_report
    task["outputs"]["owner_preparation_notice"] = owner_notice
    task["stage_status"]["preparation"] = "awaiting_confirm"
    return {"task": task}


def node_onboard(state: SurveyState) -> SurveyState:
    task = state["task"]
    records = task["outputs"].get("onboard_inspection_records", [])
    # 证据不足：标记待补拍/待补录/待说明
    for r in records:
        if not r.get("evidence"):
            r["evidenceStatus"] = "待补拍"
    task["outputs"]["onboard_inspection_records"] = records
    task["stage_status"]["onboard"] = (
        "awaiting_confirm" if records else "in_progress")
    return {"task": task}


def node_records(state: SurveyState) -> SurveyState:
    task = state["task"]
    records = task["outputs"].get("onboard_inspection_records", [])
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    invalid = []
    for r in records:
        by_category.setdefault(r.get("category", "未分类"), []).append(r)
        if not r.get("regulationRef") or not r.get("evidence"):
            invalid.append(r.get("id"))
    checklist = task["outputs"].get("base_checklist", [])
    done_items = {r.get("itemNo") for r in records if r.get("itemNo")}
    task["outputs"]["inspection_item_execution_report"] = {
        "totalItems": len(checklist),
        "executedItems": len(done_items),
        "byCategory": {k: len(v) for k, v in by_category.items()},
        "invalidRecords": invalid,  # 校验不通过的记录
    }
    changes = task["outputs"].get("new_equipment_changes", [])
    task["outputs"]["new_equipment_changes"] = changes
    task["stage_status"]["records"] = "awaiting_confirm"
    return {"task": task}


def node_certificate(state: SurveyState) -> SurveyState:
    task = state["task"]
    certs = task["outputs"].get("ship_profile_summary", {}).get(
        "certificates", {})
    records = task["outputs"].get("onboard_inspection_records", [])
    issues = [r for r in records if r.get("type") == "issue"]
    regs = task["outputs"].get("priority_checklist", {}).get(
        "regulationUpdates", [])
    memo_lines = [f"- {r.get('description')}（依据 {r.get('regulationRef', '待补')}，"
                  f"定性：{r.get('qualification', '待定性')}）" for r in issues]
    task["outputs"]["survey_memorandum"] = "\n".join(
        ["【检验备忘录】"] + (memo_lines or ["无现场问题记录"]))
    cert_ok = bool(certs.get("issuable"))
    task["outputs"]["certificate_report"] = {
        "template": "年度检验报告（CCS 标准模板）",
        "issuesCount": len(issues),
        "certificateCheck": "通过" if cert_ok else "不通过",
        "regulationBasis": [r["ref"] for r in regs],
        "draftStatus": "draft",
    }
    # 证书校验不通过：阻断自动归档
    task["outputs"]["archive_status"] = (
        "pending_review" if cert_ok else "blocked_certificate_check")
    task["stage_status"]["certificate"] = (
        "awaiting_confirm" if cert_ok else "blocked")
    return {"task": task}


def build_survey_graph():
    g = StateGraph(SurveyState)
    for stage in STAGES:
        g.add_node(stage, globals()[f"node_{stage}"])
    g.add_edge(START, "profile")
    for a, b in zip(STAGES, STAGES[1:]):
        g.add_edge(a, b)
    g.add_edge(STAGES[-1], END)
    return g.compile()


SURVEY_GRAPH = build_survey_graph()


# ---------------- 任务存储与推进 ----------------
def _task_path(task_id: str) -> str:
    return os.path.join(SURVEYS_DIR, f"{task_id}.json")


def save_task(task: Dict[str, Any]) -> None:
    task["updatedAt"] = datetime.now().isoformat()
    with open(_task_path(task["id"]), "w", encoding="utf-8") as f:
        json.dump(task, f, ensure_ascii=False, indent=2)


def load_task(task_id: str) -> Optional[Dict[str, Any]]:
    path = _task_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_tasks() -> List[Dict[str, Any]]:
    tasks = []
    for name in sorted(os.listdir(SURVEYS_DIR)):
        if name.endswith(".json"):
            try:
                with open(os.path.join(SURVEYS_DIR, name),
                          encoding="utf-8") as f:
                    tasks.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return tasks


def create_task(inputs: Dict[str, Any]) -> Dict[str, Any]:
    task = {
        "id": str(uuid.uuid4())[:8],
        "skill": "ship-annual-survey",
        "input": {
            "ship_identifier": inputs.get("ship_identifier", ""),
            "inspection_work_order": inputs.get(
                "inspection_work_order",
                f"WO-{datetime.now():%Y%m%d}-{str(uuid.uuid4())[:4]}"),
            "inspection_type": "annual-survey",
            "certificate_scope": inputs.get("certificate_scope", "全部法定证书"),
            "planned_boarding_time": inputs.get("planned_boarding_time", ""),
            "surveyor": inputs.get("surveyor", ""),
            "branch": inputs.get("branch", ""),
        },
        "task_status": "in_progress",
        "current_stage": "profile",
        "stage_status": {s: "pending" for s in STAGES},
        "confirmations": {k: {"name": v, "confirmed": False, "at": None}
                          for k, v in CONFIRM_POINTS.items()},
        "outputs": {"onboard_inspection_records": [],
                    "new_equipment_changes": []},
        "createdAt": datetime.now().isoformat(),
    }
    save_task(task)
    return task


def _gate_open(task: Dict[str, Any], stage: str) -> List[str]:
    """返回该阶段未完成的确认点名单（空 = 可进入下一阶段）"""
    return [CONFIRM_POINTS[p] for p in STAGE_GATE.get(stage, [])
            if not task["confirmations"][p]["confirmed"]]


async def _run_stage(task: Dict[str, Any], stage: str) -> Dict[str, Any]:
    node = globals()[f"node_{stage}"]
    result = node({"task": task})
    if hasattr(result, "__await__"):
        result = await result
    return result["task"]


async def advance_task(task_id: str) -> Dict[str, Any]:
    """执行当前阶段；若阶段已完成且确认点齐备，则进入下一阶段执行"""
    task = load_task(task_id)
    if task is None:
        return {"error": "任务不存在"}
    stage = task["current_stage"]
    status = task["stage_status"].get(stage)
    if status in ("done", "attention", "awaiting_confirm"):
        pending = _gate_open(task, stage)
        if pending:
            return {"task": task, "blockedBy": pending}
        idx = STAGES.index(stage)
        if idx + 1 >= len(STAGES):
            _try_complete(task)
            save_task(task)
            return {"task": task}
        stage = STAGES[idx + 1]
        task["current_stage"] = stage
    task = await _run_stage(task, stage)
    if task["stage_status"].get(stage) in ("done", "attention") and \
            not STAGE_GATE.get(stage):
        pass  # 无确认点的阶段等待下次 advance 进入下一阶段
    save_task(task)
    return {"task": task}


def confirm_point(task_id: str, point: str,
                  confirmed: bool = True) -> Dict[str, Any]:
    task = load_task(task_id)
    if task is None or point not in CONFIRM_POINTS:
        return {"error": "任务或确认点不存在"}
    task["confirmations"][point]["confirmed"] = confirmed
    task["confirmations"][point]["at"] = datetime.now().isoformat()
    stage = task["current_stage"]
    if confirmed and point == "owner_notice_send":
        mail = registry.get("send_email").run(
            to="shipowner@example.com",
            subject=f"年度检验准备通知 - {task['input']['ship_identifier']}",
            body=task["outputs"].get("owner_preparation_notice", ""))
        task["outputs"]["email_tracking_status"] = mail.get("tracking", {})
    if confirmed and point == "equipment_changes":
        # 经验船师确认后回写业务数据库；未确认不得直接回写
        registry.get("update_business_database").run(
            table="ship_equipment",
            payload={"changes": task["outputs"].get(
                "new_equipment_changes", [])})
    if confirmed and not _gate_open(task, stage) and \
            task["stage_status"].get(stage) == "awaiting_confirm":
        task["stage_status"][stage] = "done"
    if confirmed and point == "final_archive":
        _try_complete(task)
    save_task(task)
    return {"task": task}


def add_onboard_record(task_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """APP 回传的语音/照片/文字记录 → 规范问题记录"""
    task = load_task(task_id)
    if task is None:
        return {"error": "任务不存在"}
    ship = task["input"]["ship_identifier"]
    regs = registry.get("search_maritime_regulations").run(
        query=record.get("description", ""))
    normalized = {
        "id": str(uuid.uuid4())[:8],
        "type": record.get("type", "issue"),
        "itemNo": record.get("itemNo"),
        "equipment": record.get("equipment", ""),
        "description": record.get("description", ""),
        "evidence": record.get("evidence", []),
        "regulationRef": regs[0]["ref"] if regs else None,
        "qualification": record.get("qualification", "待定性"),
        "ship": ship,
        "recordedAt": datetime.now().isoformat(),
    }
    if not normalized["evidence"]:
        normalized["evidenceStatus"] = "待补拍"
    task["outputs"]["onboard_inspection_records"].append(normalized)
    if record.get("newEquipment"):
        # 设备不一致：生成人工确认清单，不得直接回写
        task["outputs"]["new_equipment_changes"].append({
            "equipment": record.get("equipment", ""),
            "change": record.get("newEquipment"),
            "confirmed": False,
        })
    save_task(task)
    return {"task": task, "record": normalized}


def _try_complete(task: Dict[str, Any]) -> None:
    """完成条件：检查项完整、问题有证据、法规可追溯、设备变更已确认、
    报告已审核且归档成功，才置 completed"""
    outputs = task["outputs"]
    records = outputs.get("onboard_inspection_records", [])
    issues = [r for r in records if r.get("type") == "issue"]
    conditions = {
        "检查项状态完整": bool(outputs.get("base_checklist")),
        "问题均有证据": all(r.get("evidence") for r in issues),
        "法规可追溯": all(r.get("regulationRef") for r in issues),
        "设备变更已确认": task["confirmations"]["equipment_changes"]["confirmed"],
        "报告已审核": task["confirmations"]["certificate_opinion"]["confirmed"],
        "归档已确认": task["confirmations"]["final_archive"]["confirmed"],
        "证书校验通过": outputs.get("archive_status") != "blocked_certificate_check",
    }
    task["completion_conditions"] = conditions
    if all(conditions.values()):
        outputs["archive_status"] = "archived"
        task["task_status"] = "completed"
        task["stage_status"]["certificate"] = "done"
        registry.get("update_business_database").run(
            table="survey_archive", payload={"taskId": task["id"]})
    else:
        task["task_status"] = "in_progress"
