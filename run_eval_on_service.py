#!/usr/bin/env python3
"""
对真实模型服务做技术指标评测：
调用现有 AI 对话流式接口 POST /api/ai/chat/classify，把每次请求/SSE 响应
记录为 OpenTelemetry GenAI 格式的 Trace 日志，再用 eval_prototype 评测。

用法:
  python run_eval_on_service.py                          # 使用内置用例，服务地址默认 http://127.0.0.1:8000
  python run_eval_on_service.py --base http://IP:8000    # 指定服务地址
  python run_eval_on_service.py --base http://IP:8000 --auth "Bearer xxx"
  python run_eval_on_service.py --cases my_cases.json    # 自定义测试用例

输出:
  eval_out/trace_<n>.json / declaration_<n>.json         # 每个用例的日志与证据说明表
  eval_out/service_eval_report.json                      # 汇总评测报告
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone

import httpx

import eval_prototype as ep

# ==================== 内置测试用例（针对船检智能体服务） ====================
DEFAULT_CASES = [
    {
        "name": "问题记录",
        "request": {
            "taskId": 10001, "clientType": "pc", "pageCode": "inspection",
            "sessionType": "operation",
            "content": "发现船体外板存在裂纹，严重程度高，位置在船首左侧外板",
        },
        "expect_action": "RECORD_ISSUE",
    },
    {
        "name": "确认整改遗留问题",
        "request": {
            "taskId": 10001, "clientType": "pc", "pageCode": "legacy",
            "sessionType": "operation",
            "content": "确认这些遗留问题已经整改",
            "actionParams": {"issueIds": [1001, 1002], "confirmNote": "确认整改完成"},
        },
        "expect_action": "CONFIRM_RECTIFICATION_ISSUES",
    },
    {
        "name": "查看检查项概览",
        "request": {
            "taskId": 10001, "clientType": "pc", "pageCode": "inspection",
            "sessionType": "operation",
            "content": "查看检查项概览",
        },
        "expect_action": "VIEW_CHECK_ITEMS_OVERVIEW",
    },
    {
        "name": "开始检验",
        "request": {
            "clientType": "pc", "pageCode": "home", "sessionType": "operation",
            "content": "开始检验",
        },
        "expect_action": "START_INSPECTION",
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_sse(text: str):
    """解析 SSE 文本为 (event, data_dict|str) 列表"""
    events = []
    for block in text.split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if event and data is not None:
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                pass
            events.append((event, data))
    return events


def call_service(base: str, auth: str, payload: dict, timeout: float = 60.0):
    url = f"{base.rstrip('/')}/api/ai/chat/classify"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if auth:
        headers["Authorization"] = auth
    started = now_iso()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.text
    ended = now_iso()
    return started, ended, parse_sse(body)


def build_trace(case: dict, base: str, started: str, ended: str, events) -> tuple:
    """把一次 SSE 调用转换为 OTel GenAI Trace + 参赛者证据说明表"""
    trace_id = uuid.uuid4().hex
    root_id = uuid.uuid4().hex[:16]
    req = case["request"]
    host = base.split("//")[-1]
    address, _, port = host.partition(":")

    start_meta = next((d for e, d in events if e == "message_start" and isinstance(d, dict)), {})
    end_meta = next((d for e, d in events if e == "message_end" and isinstance(d, dict)), {})
    deltas = [d for e, d in events if e == "answer_delta"]
    action_results = [d for e, d in events if e == "action_result" and isinstance(d, dict)]
    first_action = next((d for d in deltas if isinstance(d, dict) and d.get("actionCode")), None)

    answer_text = "\n".join(
        d.get("content", "") if isinstance(d, dict) else str(d) for d in deltas)
    status_ok = end_meta.get("status") in (None, "success") and (
        not isinstance(first_action, dict) or first_action.get("status") != "failed")

    spans = [{
        "trace_id": trace_id,
        "span_id": root_id,
        "parent_span_id": None,
        "name": "invoke_agent chat-classify",
        "start_time": started,
        "end_time": ended,
        "status.code": 1 if status_ok else 2,
        "error.type": None if status_ok else "ACTION_FAILED",
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.request.model": os.getenv("EVAL_MODEL_NAME", "ship-inspection-agent"),
        "gen_ai.response.model": os.getenv("EVAL_MODEL_NAME", "ship-inspection-agent"),
        "server.address": address,
        "server.port": int(port) if port.isdigit() else None,
        "gen_ai.conversation.id": str(start_meta.get("sessionId") or ""),
        "gen_ai.response.id": str(start_meta.get("requestId") or ""),
        "gen_ai.input.messages": [
            {"role": "user", "parts": [{"type": "text", "content": req.get("content", "")}]},
        ],
        "gen_ai.output.messages": [
            {"role": "assistant", "finish_reason": "stop",
             "parts": [{"type": "text", "content": answer_text}]},
        ],
    }]

    # 首个业务 answer_delta（action_result 型）→ execute_tool Span
    tool_span_id = None
    if isinstance(first_action, dict):
        tool_span_id = uuid.uuid4().hex[:16]
        action = first_action.get("actionCode", "")
        failed = first_action.get("status") in ("failed",)
        spans.append({
            "trace_id": trace_id,
            "span_id": tool_span_id,
            "parent_span_id": root_id,
            "name": f"execute_tool {action}",
            "start_time": started,
            "end_time": ended,
            "status.code": 2 if failed else 1,
            "error.type": "ACTION_FAILED" if failed else None,
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": action,
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": f"call-{trace_id[:8]}",
            "gen_ai.tool.call.arguments": {
                "taskId": req.get("taskId"),
                "content": req.get("content"),
                **(req.get("actionParams") or {}),
            },
            "gen_ai.tool.call.result": first_action,
        })
        # 根 Span 输入补充 tool_call 轨迹（多源信息）
        spans[0]["gen_ai.input.messages"].append({
            "role": "assistant",
            "parts": [{"type": "tool_call_request", "id": f"call-{trace_id[:8]}",
                       "name": action,
                       "arguments": spans[1]["gen_ai.tool.call.arguments"]}],
        })
        spans[0]["gen_ai.input.messages"].append({
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": f"call-{trace_id[:8]}",
                       "content": first_action}],
        })

    action_code = (first_action or {}).get("actionCode") if isinstance(first_action, dict) else None
    declaration = {
        "submission_task_id": f"CASE-{case['name']}",
        "track_name": "水运",
        "team_id": "ship-inspection-agent",
        "trace_id": trace_id,
        "task_description": req.get("content", ""),
        "deliverable_reference": "SSE answer_delta 首帧业务结果",
        "deliverable_hash": "",
        "declared_stages": (
            [{"stage_name": "接收任务并返回结果", "span_ids": [root_id]}]
            + ([{"stage_name": f"执行动作 {action_code}", "span_ids": [tool_span_id]}]
               if tool_span_id else [])
        ),
        "parallel_stages": [],
        "delivery_stage": "接收任务并返回结果",
        "declared_tools_or_skills": ([{
            "tool_or_skill_name": action_code,
            "purpose": (first_action or {}).get("actionName", ""),
            "required_params": ["content"],
            "return_type": "object",
            "failure_return": "status=failed 的 action_result",
            "external_write": True,
        }] if action_code else []),
        "memory_group_id": "",
        "related_trace_ids": [],
        "final_output_span": root_id,
    }
    trace = {"trace_id": trace_id, "spans": spans}
    meta = {
        "case": case["name"],
        "expect_action": case.get("expect_action"),
        "actual_action": action_code,
        "action_matched": (case.get("expect_action") is None
                           or case.get("expect_action") == action_code),
        "first_delta_status": (first_action or {}).get("status") if isinstance(first_action, dict) else None,
        "sse_events": [e for e, _ in events],
        "action_result_events": len(action_results),
    }
    return trace, declaration, meta


def evaluate_trace(trace: dict, decl: dict):
    results = [
        ep.evaluate_p1(trace),
        ep.evaluate_p2_static(trace),
        ep.evaluate_e1(trace, decl),
        ep.evaluate_e2(trace, decl),
        ep.evaluate_e3(trace, decl),
        ep.evaluate_l1(trace),
    ]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.getenv("EVAL_SERVICE_BASE", "http://127.0.0.1:8000"))
    parser.add_argument("--auth", default=os.getenv("EVAL_SERVICE_AUTH", ""))
    parser.add_argument("--cases", default="")
    args = parser.parse_args()

    cases = DEFAULT_CASES
    if args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases = json.load(f)

    os.makedirs("eval_out", exist_ok=True)
    all_case_reports = []
    print(f"服务地址: {args.base}   用例数: {len(cases)}")

    for i, case in enumerate(cases, 1):
        print(f"\n>>> 用例{i}: {case['name']}  content={case['request'].get('content')!r}")
        try:
            started, ended, events = call_service(args.base, args.auth, case["request"])
        except Exception as e:
            print(f"    调用失败: {e}")
            all_case_reports.append({"case": case["name"], "error": str(e)})
            continue
        trace, decl, meta = build_trace(case, args.base, started, ended, events)

        with open(f"eval_out/trace_{i}.json", "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        with open(f"eval_out/declaration_{i}.json", "w", encoding="utf-8") as f:
            json.dump(decl, f, ensure_ascii=False, indent=2)

        results = evaluate_trace(trace, decl)
        mark = "✅" if meta["action_matched"] else "❌"
        print(f"    动作识别: 预期={meta['expect_action']} 实际={meta['actual_action']} {mark}"
              f"  首帧status={meta['first_delta_status']}  SSE事件={meta['sse_events']}")
        for r in results:
            print(f"    {r.name:<26} {r.score:>6.1f}  {r.detail}")
        all_case_reports.append({
            "meta": meta,
            "metrics": [ep.asdict(r) for r in results],
            "trace_file": f"eval_out/trace_{i}.json",
            "declaration_file": f"eval_out/declaration_{i}.json",
        })

    # 汇总
    metric_sums, metric_counts = {}, {}
    matched = 0
    ok_cases = [c for c in all_case_reports if "metrics" in c]
    for c in ok_cases:
        if c["meta"]["action_matched"]:
            matched += 1
        for m in c["metrics"]:
            if m["score"] is not None:
                metric_sums[m["name"]] = metric_sums.get(m["name"], 0) + m["score"]
                metric_counts[m["name"]] = metric_counts.get(m["name"], 0) + 1
    averages = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}

    print("\n" + "=" * 70)
    print("📊 模型服务评测汇总")
    print("=" * 70)
    print(f"用例: {len(ok_cases)}/{len(cases)} 调用成功，"
          f"动作识别正确 {matched}/{len(ok_cases)}")
    for name, avg in averages.items():
        print(f"  {name:<28} 平均 {avg:.1f}")
    overall = sum(averages.values()) / len(averages) if averages else 0
    print(f"🏆 静态指标总均分: {overall:.1f}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "service_base": args.base,
        "cases": all_case_reports,
        "metric_averages": averages,
        "action_match_rate": matched / len(ok_cases) if ok_cases else 0,
        "overall_static_average": overall,
    }
    with open("eval_out/service_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("💾 汇总报告已保存至 eval_out/service_eval_report.json")


if __name__ == "__main__":
    main()
