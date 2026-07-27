"""Skill 层：可复用工作流（模型按需加载，加载时才进上下文）。

每个 Skill 是一段业务经验/流程指引，主 agent 在决策时可按需加载。
业务规则（原 harness 的流程先验）在这里作为辅助知识存在。
"""

from typing import Dict, List


class Skill:
    def __init__(self, name: str, summary: str, body: str):
        self.name = name
        self.summary = summary  # 进索引（常驻 System Prompt）
        self.body = body        # 加载时才进上下文


SKILLS: Dict[str, Skill] = {}


def register(skill: Skill) -> None:
    SKILLS[skill.name] = skill


def load(name: str) -> str:
    s = SKILLS.get(name)
    return s.body if s else ""


def prompt_index() -> str:
    lines = ["可加载的 Skill（按需引用其中经验）："]
    for s in SKILLS.values():
        lines.append(f"- {s.name}: {s.summary}")
    return "\n".join(lines)


# ---- 内置船检业务 Skill（业务规则作为辅助知识） ----
register(Skill(
    "inspection_workflow",
    "船检标准作业流程与动作先后顺序",
    """船检任务标准流程：
1. BEGIN_TASK（准备船只任务）→ GENERATE_PREPARATION_FORM（开检准备单）→ OPEN_UPLOAD_MATERIAL（上传资料）
2. START_INSPECTION（开始检验）→ ADD_CHECK_ITEM/DELETE_CHECK_ITEM → SAVE_CHECK_ITEMS → VIEW_CHECK_ITEMS_OVERVIEW
3. 检验中发现问题 → RECORD_ISSUE → CONFIRM_RECTIFICATION_ISSUES 或 PENDING_RECTIFICATION_ISSUES
4. 收尾：GENERATE_RA_REPORT → GENERATE_WORK_LOG → COMPLETE_INSPECTION_TASK
推荐下一步时优先遵循该顺序。""",
))

register(Skill(
    "intent_hints",
    "意图识别的业务提示词典",
    """常见表述映射：
- "开始准备X船只任务" → BEGIN_TASK（X 为船名）
- "开始检验/开检" → START_INSPECTION
- "发现…裂纹/松动/锈蚀/问题/缺陷/隐患" → RECORD_ISSUE
- "整理/汇总/查看…问题/检查项" → VIEW_CHECK_ITEMS_OVERVIEW
- "确认…已整改" → CONFIRM_RECTIFICATION_ISSUES；"未整改" → PENDING_RECTIFICATION_ISSUES
- "完成/结束检验任务" → COMPLETE_INSPECTION_TASK
- 闲聊、咨询、解释类问题 → GENERAL_QA""",
))
