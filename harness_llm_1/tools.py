"""Tool Call 层：原子操作。

每个工具是主 agent 可调用的最小动作单元：
  - 15 个船检业务动作（转发主服务 /api/ai/chat/classify，附显式 actionCode）
  - 大盘数据工具（指标汇总、动作分布、时间线）
工具结果回到主上下文，由主 agent 决定下一步。
"""

from typing import Any, Callable, Dict, List

# 船检业务动作（与主服务动作码一一对应）
BUSINESS_ACTIONS: Dict[str, str] = {
    "BEGIN_TASK": "船只任务准备：开始准备某船只的任务",
    "START_INSPECTION": "开始检验：启动检验任务",
    "ADD_CHECK_ITEM": "新增检查项",
    "DELETE_CHECK_ITEM": "删除检查项",
    "SAVE_CHECK_ITEMS": "保存检查项",
    "VIEW_CHECK_ITEMS_OVERVIEW": "查看检查项概览/统计/整理问题清单",
    "GENERATE_PREPARATION_FORM": "生成开检准备单",
    "GENERATE_RA_REPORT": "生成RA报告（风险评估报告）",
    "OPEN_UPLOAD_MATERIAL": "打开上传资料入口",
    "GENERATE_WORK_LOG": "生成工作日志",
    "RECORD_ISSUE": "记录问题/缺陷/隐患",
    "CONFIRM_RECTIFICATION_ISSUES": "确认遗留问题已整改",
    "PENDING_RECTIFICATION_ISSUES": "标记遗留问题未整改",
    "COMPLETE_INSPECTION_TASK": "完成检验任务（收尾归档）",
    "GENERAL_QA": "通用问答：以上都不匹配时由主服务大模型自由回答",
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
    "GENERAL_QA": "通用问答",
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


class Tool:
    """原子工具：name + 描述 + 执行函数（结果进入主上下文）"""

    def __init__(self, name: str, description: str, func: Callable[..., Any]):
        self.name = name
        self.description = description
        self.func = func

    def run(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str):
        def deco(func: Callable[..., Any]):
            self._tools[name] = Tool(name, description, func)
            return func
        return deco

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return list(self._tools)

    def prompt_index(self) -> str:
        """给主 agent 的工具索引（进 System Prompt）"""
        lines = ["可用业务动作（dispatch_action 的 action_code 取值）："]
        for code, desc in BUSINESS_ACTIONS.items():
            lines.append(f"- {code}: {desc}")
        lines.append("")
        lines.append("可用数据工具：")
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)


registry = ToolRegistry()


def build_dashboard_tools(state_getter: Callable[[], Dict[str, Any]]) -> None:
    """注册大盘数据工具（主 agent 可查询后决定大盘展示内容）"""

    @registry.register("get_metrics", "获取运行指标：总交互、成功率、未识别数、负反馈数")
    def get_metrics() -> Dict[str, Any]:
        st = state_getter()
        inter = st.get("interactions", [])
        total = len(inter)
        success = sum(1 for i in inter if i.get("status") == "success")
        return {
            "totalInteractions": total,
            "successRate": round(success / total, 3) if total else None,
            "unmatchedCount": sum(1 for i in inter if not i.get("actionCode")),
            "negativeFeedback": sum(
                1 for f in st.get("feedback", []) if f.get("rating") == "down"),
        }

    @registry.register("get_action_distribution", "获取各动作使用次数分布")
    def get_action_distribution() -> Dict[str, int]:
        st = state_getter()
        dist: Dict[str, int] = {}
        for i in st.get("interactions", []):
            code = i.get("actionCode") or "UNMATCHED"
            dist[code] = dist.get(code, 0) + 1
        return dist

    @registry.register("get_recent_interactions", "获取最近 N 条交互记录")
    def get_recent_interactions(n: int = 10) -> List[Dict[str, Any]]:
        st = state_getter()
        return st.get("interactions", [])[-n:]
