# LangGraph 自动执行版说明

`main_langgraph.py` 是在原版 `main.py` 基础上的升级版：接口契约（`POST /api/chat` 的请求体和 SSE 事件格式）**完全不变**，前端 `static/index.html` 零改动，但后端由"单纯转发大模型"升级为 **LangGraph Agent**——大模型可以自动决定调用工具、拿到结果后继续思考，循环直到完成任务。

## 运行

```bash
pip install -r requirements.txt
python main_langgraph.py
# 浏览器打开 http://localhost:8000
```

环境变量（可选）：`UPSTREAM_BASE_URL`（默认 DeepSeek）、`UPSTREAM_API_KEY`、`UPSTREAM_MODEL`。

## 工作原理

```
START ─▶ agent(LLM 决策)
            │  有 tool_calls?
            ├── 是 ─▶ tools(ToolNode 自动执行工具) ─▶ 回到 agent
            └── 否 ─▶ END（直接回答）
```

- `agent` 节点：把对话历史交给绑定了工具的 LLM（`llm.bind_tools`），LLM 输出回答或工具调用请求
- `tools` 节点：`ToolNode` 自动执行 LLM 请求的工具，把结果作为消息塞回状态
- 条件边 `should_continue`：LLM 还想调工具就循环，否则结束
- 流式输出：`graph.astream_events` 事件流中，`on_chat_model_stream` 的增量 token 按原格式 `data: {"content"/"reasoning": ...}` 发给前端；工具执行的开始/结果以 `reasoning` 事件展示在灰色思考区，让用户看到"自动执行"过程

## 内置示例工具（`TOOLS`）

| 工具 | 功能 | 触发示例问题 |
|---|---|---|
| `calculator` | 计算数学表达式 | "帮我算 (3+5)*2/7" |
| `get_current_time` | 查当前时间 | "现在几点了？" |
| `query_employee` | 模拟查询员工信息（演示对接内部系统） | "查一下张三是哪个部门的" |

多步自动执行示例："查一下张三的部门，再告诉我现在的时间"——LLM 会先后自动调用两个工具再汇总回答。

## 如何加自己的工具（如调 Java 后端）

写一个带 `@tool` 装饰器和 docstring 的函数（docstring 是 LLM 判断何时调用的依据），加入 `TOOLS` 列表即可：

```python
import httpx
from langchain_core.tools import tool

@tool
def get_order(order_id: str) -> str:
    """根据订单号查询订单详情。"""
    resp = httpx.get(f"http://your-java-backend:8080/api/orders/{order_id}", timeout=10)
    return resp.text

TOOLS = [calculator, get_current_time, query_employee, get_order]
```

## 服务端记忆（可选）

当前多轮历史由前端全量传入。若想改为服务端保存，可用 checkpointer：

```python
from langgraph.checkpoint.memory import MemorySaver
graph = builder.compile(checkpointer=MemorySaver())
# 调用时传 config={"configurable": {"thread_id": "会话id"}}，inputs 里只放最新一条用户消息
```

## 注意

- 需要模型支持 function calling（DeepSeek 支持）
- DeepSeek 账户需有余额，否则报 402 Insufficient Balance
