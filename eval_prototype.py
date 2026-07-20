#!/usr/bin/env python3
"""
技术指标评测执行方案 - 自包含原型 (MVP)
运行: python eval_prototype.py
输出: 控制台打印评测报告 + 生成 result.json
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# ==================== 内置样例数据 ====================
SAMPLE_TRACE = {
    "trace_id": "abc123",
    "spans": [
        {
            "span_id": "span-root",
            "parent_span_id": None,
            "name": "invoke_agent",
            "start_time": "2026-07-20T10:00:00Z",
            "end_time": "2026-07-20T10:05:00Z",
            "status_code": 1,
            "gen_ai_operation_name": "invoke_agent",
            "gen_ai_input_messages": {
                "parts": [
                    {"type": "text", "content": "写一份报告"},
                    {"type": "blob", "content": "base64_encoded_image"},
                    {"type": "tool_call_request", "id": "t1"},
                    {"type": "tool_call_response", "id": "t1"}
                ]
            },
            "gen_ai_output_messages": {"finish_reason": "stop", "content": "报告已生成"}
        },
        {
            "span_id": "span-tool1",
            "parent_span_id": "span-root",
            "name": "execute_tool",
            "start_time": "2026-07-20T10:01:00Z",
            "end_time": "2026-07-20T10:02:00Z",
            "status_code": 1,
            "gen_ai_operation_name": "execute_tool",
            "gen_ai_tool_name": "query_database",
            "gen_ai_tool_call_id": "call-001",
            "gen_ai_tool_call_arguments": {"sql": "SELECT * FROM users"},
            "gen_ai_tool_call_result": {"rows": 100}
        },
        {
            "span_id": "span-tool2",
            "parent_span_id": "span-root",
            "name": "execute_tool",
            "start_time": "2026-07-20T10:03:00Z",
            "end_time": "2026-07-20T10:04:00Z",
            "status_code": 2,
            "error_type": "TIMEOUT",
            "gen_ai_operation_name": "execute_tool",
            "gen_ai_tool_name": "send_email",
            "gen_ai_tool_call_id": "call-002",
            "gen_ai_tool_call_arguments": {"to": "admin@test.com"},
            "gen_ai_tool_call_result": {"error": "connection refused"}
        }
    ]
}

SAMPLE_DECLARATION = {
    "team_id": "T001",
    "trace_id": "abc123",
    "declared_stages": [
        {"stage_name": "接收任务", "span_ids": ["span-root"]},
        {"stage_name": "查询数据", "span_ids": ["span-tool1"]},
        {"stage_name": "发送邮件", "span_ids": ["span-tool2"]}
    ],
    "tools": [
        {"name": "query_database", "required_params": ["sql"], "return_type": "object"},
        {"name": "send_email", "required_params": ["to"], "return_type": "string"}
    ],
    "final_output_span": "span-root"
}

# ==================== 防篡改哈希链校验 ====================
def verify_hash_chain(log_lines: List[str]) -> tuple:
    """模拟哈希链校验"""
    if not log_lines:
        return False, "日志为空"
    prev_hash = ""
    for i, line in enumerate(log_lines):
        parts = line.split("|", 1)
        if len(parts) != 2:
            return False, f"第{i+1}行格式错误"
        declared_hash, content = parts[0], parts[1]
        computed = hashlib.sha256((prev_hash + content).encode()).hexdigest()[:8]
        if computed != declared_hash:
            return False, f"第{i+1}行hash不匹配"
        prev_hash = declared_hash
    return True, f"哈希链验证通过，共{len(log_lines)}行"

# ==================== 指标评测函数 ====================
@dataclass
class MetricResult:
    name: str
    score: float
    detail: str
    evidence: Any

def evaluate_p1(trace: Dict) -> MetricResult:
    """P1: 多源信息获取 - parts类型覆盖度"""
    all_parts = []
    for span in trace.get("spans", []):
        parts = span.get("gen_ai_input_messages", {}).get("parts", [])
        all_parts.extend([p.get("type") for p in parts if p.get("type")])
    unique = set(all_parts)
    count = len(unique)
    if count >= 4:
        score, detail = 100, f"覆盖{count}种类型(>=4)"
    elif count >= 2:
        score, detail = 66, f"覆盖{count}种类型(2-3)"
    else:
        score, detail = 33 if count == 1 else 0, f"覆盖{count}种类型(<2)"
    return MetricResult("P1-多源信息获取", score, detail, {"types": list(unique)})

def evaluate_e1(trace: Dict, decl: Dict) -> MetricResult:
    """E1: 任务编排闭环 - 检查根任务、时间、状态等"""
    spans = trace.get("spans", [])
    roots = [s for s in spans if s.get("parent_span_id") is None]
    if not roots:
        return MetricResult("E1-任务闭环", 0, "无根Span", {})
    root = roots[0]
    checks = [
        root.get("gen_ai_operation_name") in ["invoke_agent", "invoke_workflow"],
        bool(root.get("start_time") and root.get("end_time")),
        bool(root.get("status_code") is not None),
        bool(root.get("gen_ai_input_messages")),
        bool(root.get("gen_ai_output_messages"))
    ]
    # 额外检查时间顺序
    try:
        st = datetime.fromisoformat(root["start_time"].replace("Z", "+00:00"))
        et = datetime.fromisoformat(root["end_time"].replace("Z", "+00:00"))
        checks.append(et > st)
    except:
        checks.append(False)
    passed = sum(checks)
    score = (passed / len(checks)) * 100
    return MetricResult("E1-任务闭环", score, f"通过{passed}/{len(checks)}项", {"root": root.get("span_id")})

def evaluate_e2(trace: Dict, decl: Dict) -> MetricResult:
    """E2: 工具调用 - 完整率 + 参数可执行率"""
    tool_spans = [s for s in trace.get("spans", []) if s.get("gen_ai_operation_name") == "execute_tool"]
    if not tool_spans:
        return MetricResult("E2-工具调用", 0, "无工具调用", {})
    declared = {t["name"]: t for t in decl.get("tools", [])}
    complete, exec_ok = 0, 0
    for s in tool_spans:
        has_all = all([
            s.get("gen_ai_tool_name"),
            s.get("gen_ai_tool_call_id"),
            s.get("gen_ai_tool_call_arguments"),
            s.get("gen_ai_tool_call_result"),
            s.get("start_time"), s.get("end_time"),
            "status_code" in s
        ])
        if has_all:
            complete += 1
        tool_name = s.get("gen_ai_tool_name")
        if tool_name in declared:
            required = declared[tool_name].get("required_params", [])
            args = s.get("gen_ai_tool_call_arguments", {})
            if all(p in args for p in required):
                exec_ok += 1
    total = len(tool_spans)
    completeness = complete / total if total else 0
    executable = exec_ok / total if total else 0
    score = (completeness * 0.5 + executable * 0.5) * 100
    return MetricResult("E2-工具调用", score, f"完整率{completeness:.0%}, 可执行率{executable:.0%}",
                        {"total": total, "complete": complete, "executable": exec_ok})

def evaluate_l1(trace: Dict) -> MetricResult:
    """L1: 反馈机制存在性"""
    process, outcome = False, False
    for s in trace.get("spans", []):
        if s.get("gen_ai_tool_call_result") or s.get("error_type"):
            process = True
        msgs = s.get("gen_ai_input_messages", {}).get("parts", [])
        for p in msgs:
            if p.get("type") == "text" and "修正" in p.get("content", ""):
                outcome = True
    if process and outcome:
        score, detail = 100, "双反馈"
    elif process or outcome:
        score, detail = 75, "单一反馈"
    else:
        score, detail = 0, "无反馈"
    return MetricResult("L1-反馈机制", score, detail, {"process": process, "outcome": outcome})

# ==================== 报告生成与输出 ====================
def generate_report(results: List[MetricResult], hash_valid: tuple) -> Dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "hash_chain_validation": {"valid": hash_valid[0], "message": hash_valid[1]},
        "metrics": [asdict(r) for r in results],
        "total_score": sum(r.score for r in results) / len(results) if results else 0
    }

def print_report(report: Dict):
    print("\n" + "="*60)
    print("📊 智能体评测报告 (原型MVP)")
    print("="*60)
    val = report["hash_chain_validation"]
    status = "✅ 通过" if val["valid"] else "❌ 失败"
    print(f"🔐 防篡改校验: {status} - {val['message']}")
    print("\n📈 各指标得分:")
    for m in report["metrics"]:
        bar = "█" * int(m["score"] / 10) + "░" * (10 - int(m["score"] / 10))
        print(f"  {m['name']:<20} {m['score']:>6.1f}  {bar}  {m['detail']}")
    print("\n" + "-"*60)
    print(f"🏆 综合总分: {report['total_score']:.1f}")
    print("="*60)
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n💾 详细结果已保存至 result.json")

# ==================== 主程序 ====================
def build_hash_chain(contents: List[str]) -> List[str]:
    """按哈希链规则生成日志行：hash = sha256(prev_hash + content)[:8]"""
    lines = []
    prev_hash = ""
    for content in contents:
        declared = hashlib.sha256((prev_hash + content).encode()).hexdigest()[:8]
        lines.append(f"{declared}|{content}")
        prev_hash = declared
    return lines

def main():
    # 模拟日志行用于哈希链校验（实际比赛中由参赛者提交）
    mock_log_lines = build_hash_chain([f"第{i+1}行日志内容" for i in range(5)])
    hash_valid = verify_hash_chain(mock_log_lines)

    results = [
        evaluate_p1(SAMPLE_TRACE),
        evaluate_e1(SAMPLE_TRACE, SAMPLE_DECLARATION),
        evaluate_e2(SAMPLE_TRACE, SAMPLE_DECLARATION),
        evaluate_l1(SAMPLE_TRACE)
    ]
    report = generate_report(results, hash_valid)
    print_report(report)

if __name__ == "__main__":
    main()
