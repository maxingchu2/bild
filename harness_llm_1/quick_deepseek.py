"""入口：命令行快速体验主 agent 决策（服务端入口见仓库根目录 harness_app.py）。

用法:
  python -m harness_llm_1.quick_deepseek "开始准备东海01船只任务"
  python -m harness_llm_1.quick_deepseek            # 交互模式
"""

import asyncio
import json
import sys

from .harness import build_system_prompt, decide_action, recommend_next
from .plugins import plugin_manager


async def run_once(content: str) -> None:
    decision = await decide_action(content, {"pageCode": "preparation",
                                             "sessionType": "operation",
                                             "lastAction": None})
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    recs = await recommend_next(decision["actionCode"], [])
    print("下一步推荐:", json.dumps(recs, ensure_ascii=False, indent=2))


async def main() -> None:
    plugin_manager.install_all()
    print(f"[harness] System Prompt {len(build_system_prompt())} 字符已就绪")
    if len(sys.argv) > 1:
        await run_once(" ".join(sys.argv[1:]))
        return
    print("输入指令（Ctrl+C 退出）：")
    while True:
        try:
            content = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if content:
            await run_once(content)


if __name__ == "__main__":
    asyncio.run(main())
