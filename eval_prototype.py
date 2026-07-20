#!/usr/bin/env python3
"""
技术指标评测执行方案 - 静态日志评测原型 (MVP v2)

依据《技术指标评测执行方案》完整规范实现，覆盖静态日志可评测的全部指标：
  感知能力: P1 多源信息获取（parts 类型覆盖 + 输出规范性）
            P2 环境信息感知（静态部分：环境要素字段存在性）
  执行能力: E1 任务编排与执行闭环（任务边界/阶段证据/成果闭环/正常收口，20/30/30/20）
            E2 工具/技能调用（证据完整率/说明一致率/参数可执行率/结果承接率，30/25/20/25 + 防重复防拼接校验）
            E3 执行进度跟踪（阶段可见率/阶段状态完整率/日志链路完整率，50/30/20）
            E4 异常处理与人工接管（动态 API + LLM-as-judge，静态评测中标记 N/A）
  记忆能力: M2 文件记忆（沉淀/更新/复用，规则可判部分）
  学习能力: L1 反馈反思（过程反馈/结果反馈存在性，0/75/100）
  防作弊  : 哈希链防篡改校验（行首 hash = sha256(prev_hash + content)[:8]）

日志字段遵循 OpenTelemetry GenAI Semantic Conventions（gen_ai.* 点分字段名）。
纯 Python 标准库实现，不依赖任何第三方框架；C1-C4、P2 动态部分、E4、M1、M3
需要调用参赛者 API 或评审 LLM，属于动态评测，本静态原型中标记为 N/A 并说明原因。

运行: python eval_prototype.py [trace.json] [declaration.json]
输出: 控制台评测报告 + result.json
"""

import json
import sys
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

# ==================== 规范定义的 parts 类型（OTel GenAI） ====================
SPEC_PART_TYPES = {
    "text", "tool_call_request", "tool_call_response", "reasoning", "blob",
    "file", "uri", "server_tool_call", "server_tool_call_response", "generic",
}

ENV_FIELD_GROUPS = {
    "运行环境": ["gen_ai.request.model", "gen_ai.response.model",
                 "gen_ai.request.temperature", "gen_ai.request.max_tokens"],
    "网络状态": ["server.address", "server.port"],
    "业务场景上下文": ["gen_ai.conversation.id", "gen_ai.conversation.compacted",
                       "gen_ai.response.id"],
}

# ==================== 内置样例数据（OTel GenAI 字段） ====================
SAMPLE_TRACE = {
    "trace_id": "0af7651916cd43dd8448eb211c80319c",
    "spans": [
        {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            "span_id": "00f067aa0ba902b7",
            "parent_span_id": None,
            "name": "invoke_agent ship-report-agent",
            "start_time": "2026-07-20T10:00:00Z",
            "end_time": "2026-07-20T10:05:00Z",
            "status.code": 1,
            "error.type": None,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.request.model": "qwen-max",
            "gen_ai.response.model": "qwen-max-2026-01-01",
            "gen_ai.request.temperature": 0.2,
            "gen_ai.request.max_tokens": 4096,
            "server.address": "5.5.5.45",
            "server.port": 8082,
            "gen_ai.conversation.id": "conversation-001",
            "gen_ai.conversation.compacted": False,
            "gen_ai.response.id": "resp-8f14e45f",
            "gen_ai.input.messages": [
                {
                    "role": "user",
                    "parts": [
                        {"type": "text", "content": "请查询用户数据并生成一份运输统计报告，按2026年上半年数据"},
                        {"type": "file", "file_id": "file-001", "mime_type": "application/pdf"},
                        {"type": "blob", "content": "aGVsbG8=", "mime_type": "image/png"},
                        {"type": "uri", "uri": "https://example.com/policy.pdf"},
                    ],
                },
                {
                    "role": "assistant",
                    "parts": [
                        {"type": "reasoning", "content": "需要先查询数据库，再汇总生成报告"},
                        {"type": "tool_call_request", "id": "call-001",
                         "name": "query_database",
                         "arguments": {"sql": "SELECT * FROM transport_stats WHERE ym BETWEEN '2026-01' AND '2026-06'"}},
                    ],
                },
                {
                    "role": "tool",
                    "parts": [
                        {"type": "tool_call_response", "id": "call-001",
                         "content": {"rows": 100, "dataset_id": "ds-2026H1"}},
                    ],
                },
            ],
            "gen_ai.output.messages": [
                {
                    "role": "assistant",
                    "finish_reason": "stop",
                    "parts": [
                        {"type": "text",
                         "content": "运输统计报告（2026年上半年）：基于数据集 ds-2026H1 共100条记录，"
                                    "货运量同比增长5%……（报告正文）报告文件已生成 report_2026H1.pdf"},
                    ],
                }
            ],
        },
        {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            "span_id": "b7ad6b7169203331",
            "parent_span_id": "00f067aa0ba902b7",
            "name": "plan generate-report",
            "start_time": "2026-07-20T10:00:10Z",
            "end_time": "2026-07-20T10:00:30Z",
            "status.code": 1,
            "gen_ai.operation.name": "plan",
            "gen_ai.output.messages": [
                {"role": "assistant", "finish_reason": "stop",
                 "parts": [{"type": "text", "content": "步骤：1.查询数据库 2.汇总数据 3.生成报告文件"}]}
            ],
        },
        {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            "span_id": "a1b2c3d4e5f60718",
            "parent_span_id": "00f067aa0ba902b7",
            "name": "execute_tool query_database",
            "start_time": "2026-07-20T10:01:00Z",
            "end_time": "2026-07-20T10:02:00Z",
            "status.code": 1,
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "query_database",
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": "call-001",
            "gen_ai.tool.call.arguments": {"sql": "SELECT * FROM transport_stats WHERE ym BETWEEN '2026-01' AND '2026-06'"},
            "gen_ai.tool.call.result": {"rows": 100, "dataset_id": "ds-2026H1"},
            "gen_ai.tool.definitions": [
                {"type": "function", "name": "query_database",
                 "parameters": {"required": ["sql"]}},
                {"type": "function", "name": "generate_report_file",
                 "parameters": {"required": ["dataset_id", "format"]}},
            ],
        },
        {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            "span_id": "c9d8e7f6a5b40312",
            "parent_span_id": "00f067aa0ba902b7",
            "name": "execute_tool generate_report_file",
            "start_time": "2026-07-20T10:03:00Z",
            "end_time": "2026-07-20T10:04:00Z",
            "status.code": 1,
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "generate_report_file",
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": "call-002",
            "gen_ai.tool.call.arguments": {"dataset_id": "ds-2026H1", "format": "pdf"},
            "gen_ai.tool.call.result": {"file": "report_2026H1.pdf", "pages": 12},
        },
        {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            "span_id": "d4c3b2a190807060",
            "parent_span_id": "00f067aa0ba902b7",
            "name": "create_memory report-file-memory",
            "start_time": "2026-07-20T10:04:10Z",
            "end_time": "2026-07-20T10:04:20Z",
            "status.code": 1,
            "gen_ai.operation.name": "create_memory",
            "gen_ai.memory.store.id": "store-001",
            "gen_ai.memory.record.id": "rec-001",
            "gen_ai.memory.record.count": 1,
            "gen_ai.memory.records": [{"record_id": "rec-001", "file_id": "file-001",
                                       "summary": "运输政策文件要点"}],
        },
        {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            "span_id": "e5f6a7b8c9d00102",
            "parent_span_id": "00f067aa0ba902b7",
            "name": "search_memory report-file-memory",
            "start_time": "2026-07-20T10:04:30Z",
            "end_time": "2026-07-20T10:04:40Z",
            "status.code": 1,
            "gen_ai.operation.name": "search_memory",
            "gen_ai.memory.store.id": "store-001",
            "gen_ai.memory.query.text": "运输政策要点",
            "gen_ai.memory.record.count": 1,
            "gen_ai.memory.records": [{"record_id": "rec-001", "file_id": "file-001"}],
        },
    ],
}

# 参赛者“任务与日志证据说明表” + “工具或技能说明表”
SAMPLE_DECLARATION = {
    "submission_task_id": "T001-TASK-01",
    "track_name": "水运",
    "team_id": "T001",
    "trace_id": "0af7651916cd43dd8448eb211c80319c",
    "task_description": "查询2026年上半年运输数据并生成统计报告文件",
    "deliverable_reference": "report_2026H1.pdf",
    "deliverable_hash": "",
    "declared_stages": [
        {"stage_name": "接收任务", "span_ids": ["00f067aa0ba902b7"]},
        {"stage_name": "任务规划", "span_ids": ["b7ad6b7169203331"]},
        {"stage_name": "查询数据", "span_ids": ["a1b2c3d4e5f60718"]},
        {"stage_name": "生成报告", "span_ids": ["c9d8e7f6a5b40312"]},
    ],
    "parallel_stages": [],
    "delivery_stage": "生成报告",
    "declared_tools_or_skills": [
        {"tool_or_skill_name": "query_database", "purpose": "查询运输统计数据库",
         "required_params": ["sql"], "return_type": "object",
         "failure_return": "error对象", "external_write": False},
        {"tool_or_skill_name": "generate_report_file", "purpose": "根据数据集生成报告文件",
         "required_params": ["dataset_id", "format"], "return_type": "object",
         "failure_return": "error对象", "external_write": False},
    ],
    "memory_group_id": "store-001",
    "related_trace_ids": [],
    "final_output_span": "00f067aa0ba902b7",
}


# ==================== 通用工具函数 ====================
def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def get_spans(trace: Dict) -> List[Dict]:
    return trace.get("spans", [])


def find_root(spans: List[Dict]) -> Optional[Dict]:
    """优先 invoke_agent/invoke_workflow，否则取无 parent 的最外层 Span"""
    for s in spans:
        if s.get("gen_ai.operation.name") in ("invoke_agent", "invoke_workflow"):
            return s
    for s in spans:
        if not s.get("parent_span_id"):
            return s
    return None


def iter_parts(messages: Any) -> List[Dict]:
    parts = []
    if isinstance(messages, dict):
        messages = [messages]
    for msg in messages or []:
        if isinstance(msg, dict):
            parts.extend(p for p in msg.get("parts", []) if isinstance(p, dict))
    return parts


def messages_text(messages: Any) -> str:
    chunks = []
    for p in iter_parts(messages):
        if p.get("type") == "text" and isinstance(p.get("content"), str):
            chunks.append(p["content"])
    return "\n".join(chunks)


def span_by_id(spans: List[Dict]) -> Dict[str, Dict]:
    return {s.get("span_id"): s for s in spans if s.get("span_id")}


# ==================== 防篡改哈希链校验 ====================
def verify_hash_chain(log_lines: List[str], salt: str = "") -> Tuple[bool, str]:
    """行首 hash = sha256(prev_hash + content)[:8]；首行 prev_hash 为动态盐值"""
    if not log_lines:
        return False, "日志为空"
    prev_hash = salt
    for i, line in enumerate(log_lines):
        parts = line.split("|", 1)
        if len(parts) != 2:
            return False, f"第{i + 1}行格式错误"
        declared_hash, content = parts
        computed = hashlib.sha256((prev_hash + content).encode()).hexdigest()[:8]
        if computed != declared_hash:
            return False, f"第{i + 1}行hash不匹配"
        prev_hash = declared_hash
    return True, f"哈希链验证通过，共{len(log_lines)}行"


def build_hash_chain(contents: List[str], salt: str = "") -> List[str]:
    lines, prev_hash = [], salt
    for content in contents:
        declared = hashlib.sha256((prev_hash + content).encode()).hexdigest()[:8]
        lines.append(f"{declared}|{content}")
        prev_hash = declared
    return lines


# ==================== 指标结果结构 ====================
@dataclass
class MetricResult:
    metric: str
    name: str
    score: Optional[float]          # 0-100；None 表示 N/A（需动态/LLM 评测）
    detail: str
    components: Dict[str, Any] = field(default_factory=dict)
    evidence: Any = None


# ==================== P1: 多源信息获取（静态） ====================
def evaluate_p1(trace: Dict) -> MetricResult:
    spans = get_spans(trace)
    # 1) parts[].type 去重种类数（仅统计规范类型），3分制
    types = set()
    for s in spans:
        for p in iter_parts(s.get("gen_ai.input.messages")):
            t = p.get("type")
            if t in SPEC_PART_TYPES:
                types.add(t)
    n = len(types)
    coverage_score = 3 if n >= 4 else (2 if n >= 2 else (1 if n == 1 else 0))

    # 2) 输出规范性检查，统计不符合规范的输出条数，3分制
    violations = []
    for s in spans:
        outputs = s.get("gen_ai.output.messages")
        if isinstance(outputs, dict):
            outputs = [outputs]
        for msg in outputs or []:
            if not isinstance(msg, dict):
                continue
            if not msg.get("finish_reason"):
                violations.append({"span": s.get("span_id"), "issue": "缺少finish_reason"})
            for p in msg.get("parts", []):
                if not isinstance(p, dict):
                    continue
                t = p.get("type")
                if t in ("tool_call_request", "server_tool_call"):
                    if not (p.get("id") and p.get("name") and p.get("arguments") is not None):
                        violations.append({"span": s.get("span_id"),
                                           "issue": f"{t}缺少id/name/arguments"})
                if t == "blob":
                    if not (p.get("content") and (p.get("mime_type") or p.get("content_type"))):
                        violations.append({"span": s.get("span_id"), "issue": "blob缺少content/content_type"})
                if t == "uri":
                    uri = p.get("uri") or p.get("content")
                    if not (isinstance(uri, str) and uri.startswith(("http://", "https://", "file://"))):
                        violations.append({"span": s.get("span_id"), "issue": "uri无效"})
    v = len(violations)
    output_score = 3 if v == 0 else (2 if v <= 2 else (1 if v <= 5 else 0))

    score = (coverage_score + output_score) / 6 * 100
    return MetricResult(
        "P1", "P1-多源信息获取", score,
        f"parts类型覆盖{n}种({coverage_score}/3分)，输出不规范{v}条({output_score}/3分)",
        components={"coverage_score": coverage_score, "output_score": output_score},
        evidence={"part_types": sorted(types), "violations": violations},
    )


# ==================== P2: 环境信息感知（静态部分） ====================
def evaluate_p2_static(trace: Dict) -> MetricResult:
    spans = get_spans(trace)
    found: Dict[str, List[str]] = {}
    for group, fields in ENV_FIELD_GROUPS.items():
        hit = [f for f in fields if any(s.get(f) is not None for s in spans)]
        if hit:
            found[group] = hit
    static_score = 3 if found else 0
    score = static_score / 3 * 100
    return MetricResult(
        "P2", "P2-环境信息感知(静态)", score,
        f"命中环境要素分组: {'、'.join(found) if found else '无'}（{static_score}/3分）；"
        "动态部分（3个API测试用例）需调用参赛者API，未计入",
        components={"static_score": static_score, "dynamic": "N/A（需API动态测试）"},
        evidence=found,
    )


# ==================== E1: 任务编排与执行闭环（静态） ====================
def evaluate_e1(trace: Dict, decl: Dict) -> MetricResult:
    spans = get_spans(trace)
    root = find_root(spans)
    if not root:
        return MetricResult("E1", "E1-任务编排与执行闭环", 0, "无根Span", evidence={})
    trace_id = trace.get("trace_id")
    id_map = span_by_id(spans)
    rst, ret = parse_time(root.get("start_time")), parse_time(root.get("end_time"))

    # ---- 1) 任务边界完整性（6项） ----
    boundary_checks = {
        "任务范围清晰": all(s.get("trace_id") in (None, trace_id) for s in spans),
        "根任务存在": root.get("gen_ai.operation.name") in ("invoke_agent", "invoke_workflow")
                      or not root.get("parent_span_id"),
        "用户任务存在": bool(messages_text(root.get("gen_ai.input.messages"))
                             or iter_parts(root.get("gen_ai.input.messages"))),
        "最终输出存在": bool(iter_parts(root.get("gen_ai.output.messages"))),
        "时间完整": bool(rst and ret and ret > rst),
        "状态完整": root.get("status.code") is not None
                    and (root.get("status.code") != 2 or root.get("error.type")),
    }
    if not boundary_checks["时间完整"]:
        boundary = 0.0  # 时间不完整时“任务边界完整性”记 0
    else:
        boundary = sum(boundary_checks.values()) / 6

    # ---- 2) 阶段证据覆盖率 ----
    declared_stages = decl.get("declared_stages", [])
    used_span_ids = set()
    stage_results = []
    for stage in declared_stages:
        valid = False
        for sid in stage.get("span_ids", []):
            s = id_map.get(sid)
            if not s or sid in used_span_ids:
                continue
            st, et = parse_time(s.get("start_time")), parse_time(s.get("end_time"))
            has_io = bool(iter_parts(s.get("gen_ai.input.messages"))
                          or iter_parts(s.get("gen_ai.output.messages"))
                          or s.get("gen_ai.tool.call.arguments") is not None
                          or s.get("gen_ai.tool.call.result") is not None
                          or s.get("gen_ai.memory.records") is not None)
            in_trace = s.get("trace_id") in (None, trace_id)
            status_ok = s.get("status.code") is not None or s.get("error.type")
            if in_trace and st and et and has_io and status_ok:
                used_span_ids.add(sid)
                valid = True
                break
        stage_results.append({"stage": stage.get("stage_name"), "stage_valid": int(valid)})
    stage_coverage = (sum(r["stage_valid"] for r in stage_results) / len(stage_results)
                      if stage_results else 0.0)

    # ---- 3) 成果闭环得分 ----
    final_text = messages_text(root.get("gen_ai.output.messages"))
    task_text = messages_text(root.get("gen_ai.input.messages"))
    tool_results = [s.get("gen_ai.tool.call.result") for s in spans
                    if s.get("gen_ai.operation.name") == "execute_tool"
                    and s.get("gen_ai.tool.call.result") is not None]
    # 成果存在
    exist = 1.0 if final_text and final_text.strip() not in ("任务已完成", "已完成") else 0.0
    # 任务回应（规则近似：最终输出与任务文本有关键词交集）
    task_kw = {w for w in task_text.replace("，", " ").replace("、", " ").split() if len(w) >= 2}
    respond = 1.0 if final_text and (not task_kw or any(k in final_text for k in
              ["报告", "结果", "统计", "完成"] + list(task_kw))) else 0.0
    # 过程承接（工具结果标识出现在后续输入/输出中）
    if tool_results:
        carried = 0
        for r in tool_results:
            tokens = [str(v) for v in (r.values() if isinstance(r, dict) else [r])
                      if isinstance(v, (str, int, float))]
            if any(t and t in final_text for t in tokens):
                carried += 1
        carry = 1.0 if carried == len(tool_results) else (0.5 if carried else 0.0)
        closure = (exist + respond + carry) / 3
        carry_detail = f"{carried}/{len(tool_results)}个工具结果被承接"
    else:
        closure = (exist + respond) / 2
        carry_detail = "无工具调用，本项不参与计算"

    # ---- 4) 任务正常收口率（5项） ----
    key_spans = [id_map[sid] for st in declared_stages for sid in st.get("span_ids", []) if sid in id_map]
    error_spans = [s for s in spans if s.get("error.type") or s.get("status.code") == 2]
    recovery_ok = True
    if error_spans:
        latest_err_end = max(filter(None, (parse_time(s.get("end_time")) for s in error_spans)), default=None)
        later_success = any(
            s.get("status.code") == 1 and parse_time(s.get("start_time"))
            and latest_err_end and parse_time(s.get("start_time")) >= latest_err_end
            for s in spans)
        declared_fail = any(k in final_text for k in ("未完成", "失败", "无法完成"))
        recovery_ok = later_success or declared_fail
    exec_ends = [parse_time(s.get("end_time")) for s in spans
                 if s is not root and parse_time(s.get("end_time"))]
    wrapup_checks = {
        "根任务及关键阶段均有结束时间": bool(ret) and all(s.get("end_time") for s in key_spans),
        "根任务最终状态非未结束": root.get("status.code") is not None,
        "最终输出在主要执行步骤之后": bool(ret) and (not exec_ends or ret >= max(exec_ends)),
        "无仍在运行的关键子Span": all(s.get("end_time") for s in spans),
        "错误后有恢复或明确说明未完成": recovery_ok,
    }
    wrapup = sum(wrapup_checks.values()) / 5

    e1 = (0.20 * boundary + 0.30 * stage_coverage + 0.30 * closure + 0.20 * wrapup) * 100
    return MetricResult(
        "E1", "E1-任务编排与执行闭环", e1,
        f"边界{boundary:.2f} 阶段覆盖{stage_coverage:.2f} 成果闭环{closure:.2f} 收口{wrapup:.2f}",
        components={
            "任务边界完整性": {"score": boundary, "checks": boundary_checks},
            "阶段证据覆盖率": {"score": stage_coverage, "stages": stage_results},
            "成果闭环得分": {"score": closure, "成果存在": exist, "任务回应": respond,
                             "过程承接": carry_detail},
            "任务正常收口率": {"score": wrapup, "checks": wrapup_checks},
        },
        evidence={"root_span": root.get("span_id")},
    )


# ==================== E2: 工具/技能调用（静态） ====================
def evaluate_e2(trace: Dict, decl: Dict) -> MetricResult:
    spans = get_spans(trace)
    root = find_root(spans) or {}
    trace_id = trace.get("trace_id")
    rst, ret = parse_time(root.get("start_time")), parse_time(root.get("end_time"))
    tool_spans = [s for s in spans if s.get("gen_ai.operation.name") == "execute_tool"]
    if not tool_spans:
        return MetricResult("E2", "E2-工具/技能调用", 0, "无可观测工具或技能调用", evidence={})

    declared = {t.get("tool_or_skill_name") or t.get("name"): t
                for t in decl.get("declared_tools_or_skills", decl.get("tools", []))}
    final_text = messages_text(root.get("gen_ai.output.messages"))

    # 防重复/防拼接校验
    call_ids = [s.get("gen_ai.tool.call.id") for s in tool_spans]
    span_ids_all = [s.get("span_id") for s in spans]
    fraud_flags: Dict[str, List[str]] = {}
    for s in tool_spans:
        flags = []
        cid = s.get("gen_ai.tool.call.id")
        if cid and call_ids.count(cid) > 1:
            flags.append("重复调用ID")
        st, et = parse_time(s.get("start_time")), parse_time(s.get("end_time"))
        if st and et and et < st:
            flags.append("结果时间早于调用开始时间")
        if s.get("trace_id") not in (None, trace_id):
            flags.append("工具Span不属于所声明的根任务")
        if s.get("span_id") and span_ids_all.count(s.get("span_id")) > 1:
            flags.append("重复Span ID")
        if flags:
            fraud_flags[s.get("span_id")] = flags

    per_call = []
    for idx, s in enumerate(tool_spans):
        cid = s.get("gen_ai.tool.call.id")
        st, et = parse_time(s.get("start_time")), parse_time(s.get("end_time"))
        fraud = s.get("span_id") in fraud_flags

        # 调用证据完整率（7项）
        complete_checks = [
            bool(s.get("gen_ai.tool.name")),
            bool(cid) and call_ids.count(cid) == 1,
            s.get("gen_ai.tool.call.arguments") is not None,
            s.get("gen_ai.tool.call.result") is not None or bool(s.get("error.type")),
            bool(st and et) and s.get("status.code") is not None,
            bool(st and et and rst and ret and rst <= st and et <= ret),
            bool(cid),  # 参数/结果/状态同属一个调用ID（同Span记录即视为关联）
        ]
        call_complete = 0 if fraud else int(all(complete_checks))

        # 调用说明一致率
        name = s.get("gen_ai.tool.name")
        d = declared.get(name)
        if fraud or not d:
            consistent = 0
        else:
            args, result = s.get("gen_ai.tool.call.arguments"), s.get("gen_ai.tool.call.result")
            args_ok = isinstance(args, dict) or d.get("required_params") in (None, [], "无结构化参数")
            rt = d.get("return_type")
            result_ok = (rt is None or result is None
                         or (rt == "object" and isinstance(result, (dict, list)))
                         or (rt == "string" and isinstance(result, str))
                         or (rt not in ("object", "string")))
            consistent = int(args_ok and result_ok)

        # 参数可执行率
        if fraud:
            executable = 0
        else:
            required = (d or {}).get("required_params") or []
            args = s.get("gen_ai.tool.call.arguments")
            if required == "无结构化参数":
                executable = int(args is not None)
            elif not isinstance(args, dict):
                executable = 0
            else:
                executable = int(all(p in args and args[p] not in (None, "") for p in required))

        # 结果承接率
        if fraud:
            carried = 0
        else:
            result = s.get("gen_ai.tool.call.result")
            later_spans = tool_spans[idx + 1:]
            tokens = ([str(v) for v in result.values() if isinstance(v, (str, int, float))]
                      if isinstance(result, dict) else
                      ([str(result)] if isinstance(result, (str, int, float)) else []))
            in_final = any(t and t in final_text for t in tokens)
            in_later_args = any(
                t and t in json.dumps(ls.get("gen_ai.tool.call.arguments") or {}, ensure_ascii=False)
                for t in tokens for ls in later_spans)
            if s.get("status.code") == 2 or s.get("error.type"):
                carried = int(in_final or in_later_args
                              or any(k in final_text for k in ("未完成", "失败")))
            else:
                carried = int(in_final or in_later_args)

        per_call.append({
            "tool": name, "call_id": cid, "call_complete": call_complete,
            "declaration_consistent": consistent, "arguments_executable": executable,
            "result_carried": carried, "fraud_flags": fraud_flags.get(s.get("span_id"), []),
        })

    total = len(per_call)
    r_complete = sum(c["call_complete"] for c in per_call) / total
    r_consistent = sum(c["declaration_consistent"] for c in per_call) / total
    r_exec = sum(c["arguments_executable"] for c in per_call) / total
    r_carried = sum(c["result_carried"] for c in per_call) / total
    e2 = (0.30 * r_complete + 0.25 * r_consistent + 0.20 * r_exec + 0.25 * r_carried) * 100
    return MetricResult(
        "E2", "E2-工具/技能调用", e2,
        f"完整率{r_complete:.0%} 一致率{r_consistent:.0%} 可执行率{r_exec:.0%} 承接率{r_carried:.0%}"
        + (f"，异常证据{len(fraud_flags)}处" if fraud_flags else ""),
        components={"调用证据完整率": r_complete, "说明一致率": r_consistent,
                    "参数可执行率": r_exec, "结果承接率": r_carried},
        evidence={"calls": per_call, "fraud_flags": fraud_flags},
    )


# ==================== E3: 执行进度跟踪（静态） ====================
def evaluate_e3(trace: Dict, decl: Dict) -> MetricResult:
    spans = get_spans(trace)
    root = find_root(spans) or {}
    trace_id = trace.get("trace_id")
    id_map = span_by_id(spans)
    rst, ret = parse_time(root.get("start_time")), parse_time(root.get("end_time"))
    span_id_counts: Dict[str, int] = {}
    for s in spans:
        sid = s.get("span_id")
        if sid:
            span_id_counts[sid] = span_id_counts.get(sid, 0) + 1

    declared_stages = decl.get("declared_stages", [])
    visible_spans: List[Dict] = []
    visible_stage_count = 0
    stage_detail = []
    for stage in declared_stages:
        stage_name = stage.get("stage_name", "")
        hit = None
        for sid in stage.get("span_ids", []):
            s = id_map.get(sid)
            if not s or span_id_counts.get(sid, 0) != 1:
                continue
            names = " ".join(str(s.get(k) or "") for k in
                             ("name", "gen_ai.operation.name", "gen_ai.tool.name"))
            if names.strip():
                hit = s
                break
        if hit is not None:
            visible_stage_count += 1
            visible_spans.append(hit)
        stage_detail.append({"stage": stage_name, "visible": hit is not None})
    total_stages = len(declared_stages) or 1
    r_visible = visible_stage_count / total_stages

    # 阶段状态完整率
    complete = 0
    for s in visible_spans:
        st, et = parse_time(s.get("start_time")), parse_time(s.get("end_time"))
        status_ok = s.get("status.code") is not None or s.get("error.type")
        no_conflict = not (s.get("status.code") == 1 and s.get("error.type"))
        if st and et and et >= st and status_ok and no_conflict:
            complete += 1
    r_status = complete / len(visible_spans) if visible_spans else 0.0

    # 日志链路完整率
    def chain_ok(s: Dict) -> bool:
        if s.get("trace_id") not in (None, trace_id):
            return False
        if span_id_counts.get(s.get("span_id"), 0) != 1:
            return False
        seen, cur = set(), s
        while cur.get("parent_span_id"):
            pid = cur["parent_span_id"]
            if pid in seen:
                return False  # 循环
            seen.add(pid)
            parent = id_map.get(pid)
            if parent is None:
                return False  # 父Span不存在
            cur = parent
        if cur.get("span_id") != root.get("span_id"):
            return False
        st, et = parse_time(s.get("start_time")), parse_time(s.get("end_time"))
        return bool(st and et and rst and ret and rst <= st and et <= ret)

    chained = sum(1 for s in visible_spans if chain_ok(s))
    r_chain = chained / len(visible_spans) if visible_spans else 0.0

    e3 = (0.50 * r_visible + 0.30 * r_status + 0.20 * r_chain) * 100
    return MetricResult(
        "E3", "E3-执行进度跟踪", e3,
        f"阶段可见率{r_visible:.0%} 状态完整率{r_status:.0%} 链路完整率{r_chain:.0%}",
        components={"阶段可见率": r_visible, "阶段状态完整率": r_status, "日志链路完整率": r_chain},
        evidence={"stages": stage_detail},
    )


# ==================== E4: 异常处理与人工接管（动态，N/A） ====================
def evaluate_e4_placeholder() -> MetricResult:
    return MetricResult(
        "E4", "E4-异常处理与人工接管", None,
        "N/A：需通过参赛者API发送6类去业务化异常指令并结合LLM-as-judge评审OTLP日志，"
        "静态日志评测无法执行（E4 = 正确处理样本数 ÷ 6 × 100）",
        components={"异常类型": ["关键信息缺失", "指令相互矛盾", "能力或工具不存在",
                                 "无授权或高风险", "诱导编造结果", "必须人工接管"]},
    )


# ==================== M2: 文件记忆（静态规则可判部分） ====================
def evaluate_m2(trace: Dict) -> MetricResult:
    spans = sorted(get_spans(trace), key=lambda s: s.get("start_time") or "")

    file_parts = []
    upload_time = None
    for s in spans:
        for p in iter_parts(s.get("gen_ai.input.messages")):
            if p.get("type") in ("file", "uri", "blob"):
                file_parts.append(p)
                upload_time = upload_time or parse_time(s.get("start_time"))
    file_ids = {p.get("file_id") for p in file_parts if p.get("file_id")}
    uris = {p.get("uri") for p in file_parts if p.get("uri")}

    def mem_spans(*ops):
        return [s for s in spans if s.get("gen_ai.operation.name") in ops
                and s.get("status.code") == 1]

    def linked_to_file(s: Dict) -> bool:
        blob = json.dumps(s.get("gen_ai.memory.records") or [], ensure_ascii=False)
        return (not (file_ids or uris)
                or any(fid in blob for fid in file_ids)
                or any(u in blob for u in uris))

    creates = [s for s in mem_spans("create_memory", "upsert_memory")
               if s.get("gen_ai.memory.store.id")
               and (s.get("gen_ai.memory.record.id") or (s.get("gen_ai.memory.record.count") or 0) > 0)
               and linked_to_file(s)
               and (not upload_time or (parse_time(s.get("start_time")) or upload_time) >= upload_time)]
    persisted = int(bool(file_parts) and bool(creates))

    create_time = parse_time(creates[0].get("start_time")) if creates else None
    store_ids = {s.get("gen_ai.memory.store.id") for s in creates}
    updates = [s for s in mem_spans("update_memory", "upsert_memory")
               if s.get("gen_ai.memory.store.id") in store_ids
               and (s.get("gen_ai.memory.record.id") or (s.get("gen_ai.memory.record.count") or 0) > 0)
               and create_time and (parse_time(s.get("start_time")) or create_time) > create_time]
    updated = int(persisted and bool(updates))

    searches = [s for s in mem_spans("search_memory")
                if s.get("gen_ai.memory.store.id") in store_ids
                and (s.get("gen_ai.memory.record.count") or 0) > 0
                and create_time and (parse_time(s.get("start_time")) or create_time) > create_time]
    root = find_root(spans) or {}
    reused = int(persisted and bool(searches) and bool(iter_parts(root.get("gen_ai.output.messages"))))

    m2 = 100 * (persisted + updated + reused) / 3
    return MetricResult(
        "M2", "M2-文件记忆(沉淀/更新/复用)", m2,
        f"memory_persisted={persisted} memory_updated={updated} memory_reused={reused}"
        "（文件哈希/内容指纹核对需原始文件，此处按日志内文件ID/URI关联判定）",
        components={"memory_persisted": persisted, "memory_updated": updated,
                    "memory_reused": reused},
        evidence={"file_ids": sorted(file_ids), "uris": sorted(uris),
                  "memory_store_ids": sorted(x for x in store_ids if x)},
    )


# ==================== L1: 反馈反思（静态） ====================
def evaluate_l1(trace: Dict) -> MetricResult:
    spans = get_spans(trace)
    process_evidence, outcome_evidence = [], []
    for s in spans:
        if s.get("gen_ai.tool.call.result") is not None:
            process_evidence.append("gen_ai.tool.call.result")
        if s.get("status.code") == 2:
            process_evidence.append("status.code")
        if s.get("error.type"):
            process_evidence.append("error.type")
        for p in iter_parts(s.get("gen_ai.input.messages")):
            if p.get("type") == "text" and any(
                    k in (p.get("content") or "") for k in ("修正", "重新", "不对", "改成", "评价")):
                outcome_evidence.append("gen_ai.input.messages")
    process, outcome = bool(process_evidence), bool(outcome_evidence)
    if process and outcome:
        score, detail = 100.0, "双反馈（过程+结果）"
    elif process or outcome:
        score, detail = 75.0, "单一反馈（%s）" % ("过程反馈" if process else "结果反馈")
    else:
        score, detail = 0.0, "无反馈"
    return MetricResult(
        "L1", "L1-反馈反思", score, detail,
        components={"feedback_exist": int(process or outcome),
                    "process_feedback": int(process), "outcome_feedback": int(outcome)},
        evidence={"evidence_fields": sorted(set(process_evidence + outcome_evidence))},
    )


# ==================== 动态/LLM 项占位 ====================
def dynamic_placeholders() -> List[MetricResult]:
    return [
        MetricResult("C1", "C1-业务意图理解", None,
                     "N/A：需LLM判定 tool_match/missing_handled（SCORE = tool_match*4 + missing_handled*3）"),
        MetricResult("C2", "C2-逻辑推演及情境推理", None,
                     "N/A：需LLM判定 correlation/evidence_support/reasoning_chain（各*2）"),
        MetricResult("C3", "C3-任务规划与重规划", None,
                     "N/A：动态评测，需API触发重规划（SCORE = replan_new_plan*3 + replan_success*3）"),
        MetricResult("C4", "C4-澄清与确认", None,
                     "N/A：动态评测，需LLM改造prompt并调用API（SCORE = ask_question*2 + args_fixed*2 + (count_ask≤2)*2）"),
        evaluate_e4_placeholder(),
        MetricResult("M1", "M1-会话与任务记忆", None,
                     "N/A：需LLM识别记忆候选并判定证据链（找到一条完整证据链 M1=100，否则 0）"),
        MetricResult("M3", "M3-用户偏好记忆", None,
                     "N/A：需跨会话日志+LLM识别长期偏好并校验证据链（找到一条完整证据链 M3=100，否则 0）"),
    ]


# ==================== 能力汇总 ====================
def aggregate(results: List[MetricResult]) -> Dict[str, Any]:
    by = {r.metric: r for r in results}

    def val(m):
        r = by.get(m)
        return r.score if r and r.score is not None else None

    # S执行 = 0.28×E1 + 0.24×E2 + 0.24×E3 + 0.24×E4（E4缺失时按可评项归一化）
    exec_weights = {"E1": 0.28, "E2": 0.24, "E3": 0.24, "E4": 0.24}
    avail = {m: val(m) for m in exec_weights if val(m) is not None}
    s_exec = (sum(exec_weights[m] * v for m, v in avail.items())
              / sum(exec_weights[m] for m in avail)) if avail else None

    perception = [v for v in (val("P1"), val("P2")) if v is not None]
    s_perc = sum(perception) / len(perception) if perception else None
    memory = [v for v in (val("M1"), val("M2"), val("M3")) if v is not None]
    s_mem = sum(memory) / len(memory) if memory else None
    s_learn = val("L1")

    return {
        "感知能力": s_perc,
        "认知能力": None,  # C1-C4 均需LLM/动态评测
        "执行能力": s_exec,
        "记忆能力": s_mem,
        "学习能力": s_learn,
        "说明": "N/A项（动态/LLM评测指标）未计入对应能力分；执行能力按可评指标权重归一化",
    }


# ==================== 报告输出 ====================
def generate_report(results: List[MetricResult], hash_valid: Tuple[bool, str],
                    trace: Dict, decl: Dict) -> Dict:
    static_scores = [r.score for r in results if r.score is not None]
    return {
        "timestamp": datetime.now().isoformat(),
        "trace_id": trace.get("trace_id"),
        "team_id": decl.get("team_id"),
        "framework": "纯Python标准库（json/hashlib/datetime/dataclasses），无第三方框架依赖",
        "hash_chain_validation": {"valid": hash_valid[0], "message": hash_valid[1]},
        "metrics": [asdict(r) for r in results],
        "capability_scores": aggregate(results),
        "static_average": sum(static_scores) / len(static_scores) if static_scores else 0,
    }


def print_report(report: Dict):
    print("\n" + "=" * 78)
    print("📊 智能体技术指标评测报告（静态日志评测原型 v2）")
    print("=" * 78)
    print(f"Trace: {report['trace_id']}   队伍: {report['team_id']}")
    val = report["hash_chain_validation"]
    print(f"🔐 防篡改校验: {'✅ 通过' if val['valid'] else '❌ 失败'} - {val['message']}")
    print("\n📈 指标得分:")
    for m in report["metrics"]:
        if m["score"] is None:
            print(f"  {m['name']:<28} {'N/A':>7}  {m['detail']}")
        else:
            bar = "█" * int(m["score"] / 10) + "░" * (10 - int(m["score"] / 10))
            print(f"  {m['name']:<28} {m['score']:>6.1f}  {bar}  {m['detail']}")
    print("\n🧭 能力汇总:")
    for k, v in report["capability_scores"].items():
        if k == "说明":
            continue
        print(f"  {k:<10} {'N/A' if v is None else f'{v:.1f}'}")
    print(f"  ({report['capability_scores']['说明']})")
    print("-" * 78)
    print(f"🏆 静态可评指标均分: {report['static_average']:.1f}")
    print("=" * 78)
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n💾 详细结果已保存至 result.json")


# ==================== 主程序 ====================
def main():
    trace, decl = SAMPLE_TRACE, SAMPLE_DECLARATION
    if len(sys.argv) >= 2:
        with open(sys.argv[1], encoding="utf-8") as f:
            trace = json.load(f)
    if len(sys.argv) >= 3:
        with open(sys.argv[2], encoding="utf-8") as f:
            decl = json.load(f)

    # 模拟带动态盐值的哈希链日志（实际比赛中由参赛者提交）
    salt = "salt-2026-07-20"
    mock_log_lines = build_hash_chain(
        [f"第{i + 1}行日志内容" for i in range(5)], salt=salt)
    hash_valid = verify_hash_chain(mock_log_lines, salt=salt)

    results = [
        evaluate_p1(trace),
        evaluate_p2_static(trace),
        evaluate_e1(trace, decl),
        evaluate_e2(trace, decl),
        evaluate_e3(trace, decl),
        evaluate_m2(trace),
        evaluate_l1(trace),
    ] + dynamic_placeholders()

    report = generate_report(results, hash_valid, trace, decl)
    print_report(report)


if __name__ == "__main__":
    main()
