"""核心调度引擎：大模型主 agent 控制一切，业务规则仅作辅助兜底。

每轮请求经 LangGraph 流水线：
  assemble（组装 System Prompt：角色+Tools+MCP+Skill索引+记忆+目标）
    → decide（DeepSeek 主 agent 决策动作）
    → assist（规则辅助：仅当 LLM 失败/低置信度时兜底）
    → finalize（触发 Hook、返回决策）
推荐与大盘分析分别由 RECOMMENDER / DASHBOARD_ANALYST 子 agent（独立上下文）完成。
"""

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from . import config, skills
from .goal import goal_manager
from .hooks import hooks
from .mcp import mcp_manager
from .memory import memory
from .subagent import DASHBOARD_ANALYST, RECOMMENDER, parse_json_loose
from .tools import ACTION_NAMES, ACTION_SAMPLE_CONTENT, BUSINESS_ACTIONS, registry

VALID_ACTIONS = set(BUSINESS_ACTIONS)

ROLE_PROMPT = (
    "你是船检智能体的主控 Agent（master agent），全权决定每条用户消息应执行的动作。"
    "你必须从可用业务动作中选择一个 action_code；无匹配业务动作时选 GENERAL_QA。"
    "用户教过的纠错记忆优先级最高。"
    '只输出 JSON：{"action_code": "...", "ship_name": "...或null", '
    '"confidence": 0到1, "reason": "一句话中文"}'
)

# 规则辅助（仅兜底，不主导）
ASSIST_PATTERNS = [
    (re.compile(r"开始准备\s*(.*?)\s*(?:的)?船只(?:的)?任务"), "BEGIN_TASK"),
    (re.compile(r"开始检验|开检$"), "START_INSPECTION"),
    (re.compile(r"发现.{0,20}(问题|缺陷|隐患|裂纹|松动|锈蚀|损坏)"), "RECORD_ISSUE"),
    (re.compile(r"完成.{0,6}检验任务"), "COMPLETE_INSPECTION_TASK"),
    (re.compile(r"(查看|整理|汇总).{0,10}(检查项|概览|问题)"), "VIEW_CHECK_ITEMS_OVERVIEW"),
]


class DecideState(TypedDict, total=False):
    content: str
    context: Dict[str, Any]
    system_prompt: str
    decision: Dict[str, Any]
    error: str


def build_system_prompt() -> str:
    """System Prompt 组装：[角色]+[Tools]+[MCP]+[Skill索引]+[记忆]+[目标]"""
    return "\n\n".join([
        ROLE_PROMPT,
        registry.prompt_index(),
        mcp_manager.prompt_index(),
        skills.prompt_index(),
        skills.load("intent_hints"),
        "跨会话记忆：\n" + memory.prompt_block(),
        goal_manager.prompt_block(),
    ])


async def _llm_json(system: str, user: str,
                    max_tokens: int = 300) -> Optional[Any]:
    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT) as client:
        resp = await client.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": config.LLM_MODEL,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0.1, "max_tokens": max_tokens},
        )
        resp.raise_for_status()
        return parse_json_loose(resp.json()["choices"][0]["message"]["content"])


# ---------------- LangGraph 节点 ----------------
def node_assemble(state: DecideState) -> DecideState:
    return {"system_prompt": build_system_prompt()}


async def node_decide(state: DecideState) -> DecideState:
    ctx = state.get("context", {})
    user = (f"页面: {ctx.get('pageCode')}，会话类型: {ctx.get('sessionType')}，"
            f"上一动作: {ctx.get('lastAction')}\n用户消息: {state['content']}")
    try:
        data = await _llm_json(state["system_prompt"], user)
        if isinstance(data, dict) and data.get("action_code") in VALID_ACTIONS:
            return {"decision": {
                "actionCode": data["action_code"],
                "shipName": data.get("ship_name") or None,
                "confidence": float(data.get("confidence") or 0),
                "reason": data.get("reason", ""),
                "decidedBy": "llm",
            }}
        return {"error": f"LLM 输出无效: {data}"}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"error": f"LLM 调用失败: {e}"}


def node_assist(state: DecideState) -> DecideState:
    """规则辅助：仅当主 agent 失败或低置信度时兜底"""
    decision = state.get("decision")
    if decision and decision.get("confidence", 0) >= 0.5:
        return {}
    content = state.get("content", "")
    for pattern, code in ASSIST_PATTERNS:
        m = pattern.search(content)
        if m:
            return {"decision": {
                "actionCode": code,
                "shipName": (m.group(1).strip()
                             if code == "BEGIN_TASK" and m.groups() else None),
                "confidence": 0.4,
                "reason": "规则辅助兜底",
                "decidedBy": "rule_assist",
            }}
    if decision:
        return {}
    return {"decision": {
        "actionCode": "GENERAL_QA", "shipName": None, "confidence": 0.2,
        "reason": "无法识别，交通用问答", "decidedBy": "fallback",
    }}


def node_finalize(state: DecideState) -> DecideState:
    decision = state["decision"]
    hooks.fire("after_decide", decision)
    if state.get("error"):
        hooks.fire("on_error", {"error": state["error"]})
    return {}


def build_decide_graph():
    g = StateGraph(DecideState)
    g.add_node("assemble", node_assemble)
    g.add_node("decide", node_decide)
    g.add_node("assist", node_assist)
    g.add_node("finalize", node_finalize)
    g.add_edge(START, "assemble")
    g.add_edge("assemble", "decide")
    g.add_edge("decide", "assist")
    g.add_edge("assist", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


DECIDE_GRAPH = build_decide_graph()


async def decide_action(content: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """主 agent 决策入口：返回 {actionCode, shipName, confidence, reason, decidedBy}"""
    hooks.fire("before_decide", {"content": content})
    result = await DECIDE_GRAPH.ainvoke({"content": content, "context": context})
    return result["decision"]


# ---------------- 子 agent 任务 ----------------
async def recommend_next(last_action: Optional[str],
                         recent: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """由推荐子 agent 生成下一步建议；失败时用流程 Skill 兜底"""
    candidates = [c for c in BUSINESS_ACTIONS if c != "GENERAL_QA"]
    task = (f"候选动作码: {candidates}\n"
            f"最近动作: {[i.get('actionCode') for i in recent[-5:]]}\n"
            f"上一动作: {last_action}\n"
            f"标准流程参考:\n{skills.load('inspection_workflow')}")
    try:
        data = parse_json_loose(await RECOMMENDER.run(task))
        recs = []
        for item in (data or []):
            code = item.get("actionCode")
            if code in VALID_ACTIONS and code != "GENERAL_QA":
                recs.append({
                    "actionCode": code,
                    "actionName": ACTION_NAMES.get(code, code),
                    "sampleContent": ACTION_SAMPLE_CONTENT.get(code, ""),
                    "reason": item.get("reason", "主agent推荐"),
                    "source": "llm",
                })
        if recs:
            return recs[:4]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        hooks.fire("on_error", {"error": f"推荐子agent失败: {e}"})
    # Skill 兜底
    fallback = ["START_INSPECTION", "BEGIN_TASK", "VIEW_CHECK_ITEMS_OVERVIEW"]
    return [{
        "actionCode": c, "actionName": ACTION_NAMES[c],
        "sampleContent": ACTION_SAMPLE_CONTENT.get(c, ""),
        "reason": "流程兜底", "source": "rule_assist",
    } for c in fallback]


async def analyze_dashboard(stats: Dict[str, Any]) -> Dict[str, Any]:
    """大盘分析子 agent：主 agent 视角总结运行状况；失败时返回空分析"""
    try:
        data = parse_json_loose(
            await DASHBOARD_ANALYST.run(json.dumps(stats, ensure_ascii=False)))
        if isinstance(data, dict):
            data["analyzedBy"] = "llm"
            return data
    except (httpx.HTTPError, ValueError, KeyError) as e:
        hooks.fire("on_error", {"error": f"大盘分析子agent失败: {e}"})
    return {"summary": "", "insights": [], "suggestions": [],
            "analyzedBy": "unavailable"}


def learn_from_feedback(utterance: str, expected_action: str) -> None:
    """负反馈 → 写入跨会话纠错记忆（下一轮 System Prompt 即生效）"""
    if expected_action in VALID_ACTIONS:
        memory.add_correction(utterance, expected_action, "用户负反馈")
        hooks.fire("on_feedback", {"rating": "down",
                                   "expectedAction": expected_action})
