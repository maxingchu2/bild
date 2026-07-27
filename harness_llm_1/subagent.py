"""SubAgent 层：独立上下文的子智能体（任务级生命周期）。

主 agent 可派生子 agent 处理独立子任务（如大盘分析、日志洞察），
子 agent 有自己的 System Prompt 和上下文，结果以摘要形式返回主上下文。
"""

import json
from typing import Any, Dict, List, Optional

import httpx

from . import config


async def _chat(messages: List[Dict[str, str]],
                temperature: float = 0.2,
                max_tokens: int = 800) -> str:
    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT) as client:
        resp = await client.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": config.LLM_MODEL, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class SubAgent:
    """独立上下文子智能体"""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    async def run(self, task: str) -> str:
        return await _chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ])


DASHBOARD_ANALYST = SubAgent(
    "dashboard-analyst",
    "你是船检智能体运营分析师。基于给定的交互统计 JSON，输出简短中文分析："
    "1) 一句话总结当前使用状况；2) 最多3条洞察（如高频动作、失败点、学习效果）；"
    "3) 最多2条给用户的建议。输出 JSON："
    '{"summary": "...", "insights": ["..."], "suggestions": ["..."]}，'
    "不要输出其他内容。",
)

RECOMMENDER = SubAgent(
    "next-action-recommender",
    "你是船检作业向导。根据用户最近的动作和标准船检流程，推荐接下来最有用的1-4个动作。"
    "只能从给定候选动作码中选择。输出 JSON 数组："
    '[{"actionCode": "...", "reason": "一句话中文理由"}]，不要输出其他内容。',
)


def parse_json_loose(text: str) -> Optional[Any]:
    """容忍 markdown 代码块等包装的 JSON 解析"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0],
                default=-1)
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    return None
