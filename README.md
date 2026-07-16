# 多轮对话助手

基于 FastAPI 的多轮对话服务，将请求转发到 OpenAI 兼容的模型接口，并以 SSE 流式返回 Markdown 内容，自带一个简单的流式测试页面。

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 8000
```

打开浏览器访问 http://localhost:8000 即可测试流式多轮对话。

## 配置（环境变量）

| 变量 | 默认值 |
|---|---|
| `UPSTREAM_URL` | `https://api.deepseek.com/v1/chat/completions` |
| `UPSTREAM_API_KEY` | （DeepSeek API Key） |
| `UPSTREAM_MODEL` | `deepseek-v4-flash` |

## 接口

### POST /api/chat

请求体：

```json
{
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "max_tokens": 2048,
  "temperature": 0.7
}
```

多轮对话时，客户端把完整历史（含 assistant 回复）放入 `messages` 一并发送。

响应为 SSE 流，每条事件形如：

```
data: {"content": "增量文本"}

data: [DONE]
```

出错时事件为 `data: {"error": "..."}`。

curl 测试：

```bash
curl -N http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}]}'
```
