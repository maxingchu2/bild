"""Goal/Plan 层：目标驱动（用户设目标，每轮注入，目标级长期生命周期）。

目标以 JSON 文件存于 output/goals/，主 agent 每轮决策时注入活跃目标，
用于让推荐和决策朝目标推进（如「本周完成东海01的年度检验」）。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from . import config


class GoalManager:
    def __init__(self, goals_dir: str = config.GOALS_DIR):
        self.goals_dir = goals_dir
        os.makedirs(goals_dir, exist_ok=True)

    def _path(self, goal_id: str) -> str:
        return os.path.join(self.goals_dir, f"{goal_id}.json")

    def set_goal(self, goal_id: str, description: str,
                 steps: List[str] | None = None) -> Dict[str, Any]:
        goal = {
            "id": goal_id,
            "description": description,
            "steps": steps or [],
            "doneSteps": [],
            "status": "active",
            "createdAt": datetime.now().isoformat(),
        }
        with open(self._path(goal_id), "w", encoding="utf-8") as f:
            json.dump(goal, f, ensure_ascii=False, indent=2)
        return goal

    def list_goals(self) -> List[Dict[str, Any]]:
        goals = []
        for name in sorted(os.listdir(self.goals_dir)):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(self.goals_dir, name),
                              encoding="utf-8") as f:
                        goals.append(json.load(f))
                except (json.JSONDecodeError, OSError):
                    continue
        return goals

    def active_goals(self) -> List[Dict[str, Any]]:
        return [g for g in self.list_goals() if g.get("status") == "active"]

    def complete_step(self, goal_id: str, step: str) -> None:
        path = self._path(goal_id)
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            goal = json.load(f)
        if step not in goal["doneSteps"]:
            goal["doneSteps"].append(step)
        if set(goal["steps"]) and set(goal["steps"]) <= set(goal["doneSteps"]):
            goal["status"] = "done"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(goal, f, ensure_ascii=False, indent=2)

    def prompt_block(self) -> str:
        goals = self.active_goals()
        if not goals:
            return "（当前无活跃目标）"
        lines = ["当前活跃目标（决策时朝目标推进）："]
        for g in goals:
            done = len(g.get("doneSteps", []))
            total = len(g.get("steps", []))
            lines.append(f"- {g['description']}（进度 {done}/{total}）")
        return "\n".join(lines)


goal_manager = GoalManager()
