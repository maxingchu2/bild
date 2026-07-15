"""多智能体冒烟测试：用假 LLM 验证 supervisor 路由 + 各智能体工具自动执行（不耗 API 额度）。"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import AIMessage, HumanMessage

import main_multi_agent as m


class FakeRouter:
    def __init__(self, target):
        self.target = target

    async def ainvoke(self, messages):
        return m.Route(next_agent=self.target)


class FakeAgentLLM:
    """第一次调用发出工具调用，第二次给出最终回答。"""

    def __init__(self, tool_name, args):
        self.tool_name = tool_name
        self.args = args
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{"name": self.tool_name, "args": self.args, "id": "c1"}])
        return AIMessage(content=f"工具结果: {messages[-1].content}")


async def run_case(router_target, tool_name, args, question):
    m.router_llm = FakeRouter(router_target)
    fake = FakeAgentLLM(tool_name, args)
    cfg = m.AGENTS[router_target]
    state = {"messages": [HumanMessage(content=question)], "next_agent": ""}
    route = await m.supervisor_node(state)
    assert route["next_agent"] == router_target, f"路由错误: {route}"
    print(f"路由 -> {route['next_agent']} OK")

    ai = await fake.ainvoke(state["messages"])
    assert ai.tool_calls
    tool = next(t for t in cfg["tools"] if t.name == tool_name)
    tool_msg = await tool.ainvoke(ai.tool_calls[0])
    print(f"工具 {tool_name} 执行结果: {tool_msg.content[:100]}")
    final = await fake.ainvoke([tool_msg])
    assert final.content
    print(f"最终回答: {final.content[:100]}")
    print("PASS\n")


async def main():
    await run_case("task_agent", "query_pending_tasks", {}, "查询所有待办任务")
    await run_case("prepare_agent", "generate_preparation_sheet", {"ship_name": "远洋之星"}, "生成远洋之星的检验前准备单")
    await run_case("archive_agent", "archive_record", {"ship_name": "东海明珠"}, "归档东海明珠的检验记录")
    print("全部用例通过")


asyncio.run(main())
