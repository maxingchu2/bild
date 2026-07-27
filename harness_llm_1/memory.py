"""Memory 层：跨会话记忆（harness 自动写，启动/每轮注入）。

存储：
  - 纠错记忆：用户负反馈教会的「说法 → 正确动作」
  - 偏好记忆：用户常用动作、称呼、习惯
持久化到 output/memory.json。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from . import config


class Memory:
    def __init__(self, path: str = config.MEMORY_FILE):
        self.path = path
        self.data: Dict[str, Any] = {"corrections": [], "preferences": []}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ---- 纠错记忆 ----
    def add_correction(self, utterance: str, action_code: str, note: str = "") -> None:
        self.data["corrections"] = [
            c for c in self.data["corrections"] if c["utterance"] != utterance
        ]
        self.data["corrections"].append({
            "utterance": utterance,
            "actionCode": action_code,
            "note": note,
            "at": datetime.now().isoformat(),
        })
        self.save()

    def corrections(self) -> List[Dict[str, Any]]:
        return self.data.get("corrections", [])

    # ---- 偏好记忆 ----
    def add_preference(self, text: str) -> None:
        if text and text not in self.data["preferences"]:
            self.data["preferences"].append(text)
            self.data["preferences"] = self.data["preferences"][-50:]
            self.save()

    def prompt_block(self) -> str:
        """注入 System Prompt 的记忆块"""
        lines = []
        if self.data.get("corrections"):
            lines.append("用户教过的纠错记忆（最高优先级，直接照做）：")
            for c in self.data["corrections"][-30:]:
                lines.append(f"- 当用户说「{c['utterance']}」→ 动作 {c['actionCode']}")
        if self.data.get("preferences"):
            lines.append("用户偏好：")
            for p in self.data["preferences"][-10:]:
                lines.append(f"- {p}")
        return "\n".join(lines) if lines else "（暂无跨会话记忆）"


memory = Memory()
