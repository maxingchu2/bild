"""Hook 层：生命周期钩子（事件触发，不进上下文，配置级常驻）。

事件：
  before_decide / after_decide / after_execute / on_feedback / on_error
用途：审计日志、指标累计、自动落盘等旁路逻辑。
"""

from typing import Any, Callable, Dict, List

HookFunc = Callable[[Dict[str, Any]], None]


class HookManager:
    def __init__(self) -> None:
        self._hooks: Dict[str, List[HookFunc]] = {}

    def on(self, event: str):
        def deco(func: HookFunc):
            self._hooks.setdefault(event, []).append(func)
            return func
        return deco

    def fire(self, event: str, payload: Dict[str, Any]) -> None:
        for func in self._hooks.get(event, []):
            try:
                func(payload)
            except Exception as e:  # 钩子失败不影响主流程
                print(f"[hook:{event}] error: {e}", flush=True)


hooks = HookManager()


@hooks.on("after_decide")
def _log_decision(payload: Dict[str, Any]) -> None:
    print(f"[harness] 主agent决策: action={payload.get('actionCode')} "
          f"conf={payload.get('confidence')} via={payload.get('decidedBy')}",
          flush=True)


@hooks.on("on_feedback")
def _log_feedback(payload: Dict[str, Any]) -> None:
    print(f"[harness] 用户反馈: rating={payload.get('rating')} "
          f"expected={payload.get('expectedAction')}", flush=True)


@hooks.on("on_error")
def _log_error(payload: Dict[str, Any]) -> None:
    print(f"[harness] 错误: {payload.get('error')}", flush=True)
