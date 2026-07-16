"""本地冒烟测试：用假 LLM 验证 LangGraph 的 agent->tools->agent 自动执行循环（不消耗 API 额度）。"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import AIMessage, HumanMessage

import main_langgraph as m


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "calculator", "args": {"expression": "(3+5)*2/7"}, "id": "call_1"}],
            )
        tool_result = messages[-1].content
        return AIMessage(content=f"计算结果是 {tool_result}")


async def main():
    m.llm_with_tools = FakeLLM()
    result = await m.graph.ainvoke({"messages": [HumanMessage(content="帮我算 (3+5)*2/7")]})
    msgs = result["messages"]
    for msg in msgs:
        print(type(msg).__name__, "->", repr(msg.content)[:80])
    assert any(type(x).__name__ == "ToolMessage" for x in msgs), "工具未被自动执行"
    assert "2.28" in msgs[-1].content
    print("PASS: 工具自动执行且最终回答正确")


asyncio.run(main())
