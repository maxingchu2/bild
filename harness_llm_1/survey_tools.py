"""ship-annual-survey Skill 所需工具集（requires_tools）。

8 个原子工具，数据源为 data/*.csv 与内置法规知识库；
真实环境可替换为业务系统接口，函数签名保持不变。
"""

import csv
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .tools import registry

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _read_csv(name: str) -> List[Dict[str, str]]:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _match_ship(row_name: str, identifier: str) -> bool:
    a, b = (row_name or "").strip(), (identifier or "").strip()
    return bool(a and b) and (a == b or a in b or b in a)


@registry.register("get_ship_profile", "调取船舶档案：船型、建造日期、检验类型、状态")
def get_ship_profile(ship_identifier: str) -> Dict[str, Any]:
    for row in _read_csv("ships.csv"):
        if _match_ship(row.get("船名", ""), ship_identifier) or \
                row.get("CCSNO", "") == ship_identifier:
            return {
                "found": True,
                "shipName": row.get("船名"),
                "ccsNo": row.get("CCSNO"),
                "shipType": row.get("船舶类型"),
                "builtDate": row.get("建造日期"),
                "surveyType": row.get("检验类型"),
                "status": row.get("状态"),
            }
    return {"found": False, "shipIdentifier": ship_identifier,
            "missing": ["船舶档案"]}


@registry.register("get_certificate_status", "查询船舶证书状态与签发条件")
def get_certificate_status(ship_identifier: str) -> Dict[str, Any]:
    profile = get_ship_profile(ship_identifier)
    if not profile.get("found"):
        return {"found": False, "missing": ["证书记录"]}
    built = profile.get("builtDate") or "2020-01-01"
    try:
        anniversary = datetime.strptime(built, "%Y-%m-%d").replace(
            year=datetime.now().year)
    except ValueError:
        anniversary = datetime.now()
    window_open = anniversary - timedelta(days=90)
    window_close = anniversary + timedelta(days=90)
    certs = [
        {"name": "船舶检验证书", "status": "有效",
         "expiry": (anniversary + timedelta(days=365 * 2)).strftime("%Y-%m-%d")},
        {"name": "国际载重线证书", "status": "有效",
         "expiry": (anniversary + timedelta(days=365 * 3)).strftime("%Y-%m-%d")},
        {"name": "货船安全构造证书", "status": "有效",
         "expiry": (anniversary + timedelta(days=365)).strftime("%Y-%m-%d")},
    ]
    return {
        "found": True,
        "shipName": profile.get("shipName"),
        "annualWindow": {
            "open": window_open.strftime("%Y-%m-%d"),
            "close": window_close.strftime("%Y-%m-%d"),
        },
        "certificates": certs,
        "issuable": all(c["status"] == "有效" for c in certs),
    }


@registry.register("generate_base_checklist", "调用检查项模型生成基础检查项")
def generate_base_checklist(ship_identifier: str) -> List[Dict[str, Any]]:
    items = []
    for row in _read_csv("inspection_items.csv"):
        if _match_ship(row.get("船名", ""), ship_identifier):
            items.append({
                "itemNo": row.get("编号"),
                "name": row.get("名称"),
                "category": row.get("类别"),
                "risk": row.get("风险"),
                "source": "base_model",
                "status": "pending",
            })
    return items


@registry.register("get_outstanding_memos", "调用遗留备忘库：未关闭问题与复核事项")
def get_outstanding_memos(ship_identifier: str) -> List[Dict[str, Any]]:
    memos = []
    for row in _read_csv("legacy_issues.csv"):
        if _match_ship(row.get("船名", ""), ship_identifier):
            memos.append({
                "memoNo": row.get("编号"),
                "issue": row.get("问题"),
                "status": row.get("状态"),
                "needRecheck": row.get("状态") != "已关闭",
            })
    return memos


@registry.register("get_ship_risk_history", "调用风险监控系统：损坏、修理与事故历史")
def get_ship_risk_history(ship_identifier: str) -> List[Dict[str, Any]]:
    history = []
    for memo in get_outstanding_memos(ship_identifier):
        history.append({
            "type": "遗留缺陷",
            "detail": memo["issue"],
            "riskLevel": "高" if "压力不足" in memo["issue"] else "中",
            "recordedAt": None,
        })
    return history


# 内置法规知识库（船检垂域，可替换为向量检索）
_REGULATIONS = [
    {"ref": "SOLAS II-1", "title": "构造-结构、分舱与稳性",
     "keywords": ["船体", "结构", "板厚", "水密"],
     "update": "2024 修正案：加强水密完整性年度核查要求"},
    {"ref": "SOLAS II-2", "title": "构造-防火、探火和灭火",
     "keywords": ["消防", "灭火", "探测", "应急消防泵"],
     "update": "2023 修正案：应急消防泵压力试验记录须留存"},
    {"ref": "SOLAS III", "title": "救生设备与装置",
     "keywords": ["救生", "救生艇", "救生圈", "降落"],
     "update": "2024 修正案：救生艇降落装置年度动态试验"},
    {"ref": "MARPOL Annex I", "title": "防止油类污染规则",
     "keywords": ["油污", "排油", "机舱"],
     "update": "2024：EGCS 排放监测抽查纳入年检"},
    {"ref": "CCS 钢质海船入级规范", "title": "入级检验-年度检验范围",
     "keywords": ["年度检验", "证书", "签注"],
     "update": "2025 版：年度检验签注需附电子证据链"},
]


@registry.register("search_maritime_regulations", "检索船检法规知识库与近期法规更新")
def search_maritime_regulations(query: str) -> List[Dict[str, Any]]:
    q = query or ""
    hits = []
    for reg in _REGULATIONS:
        if any(k in q for k in reg["keywords"]):
            hits.append(reg)
    return hits or [r for r in _REGULATIONS if r["ref"].startswith("CCS")]


@registry.register("send_email", "发送邮件（船东通知等）并返回跟踪状态")
def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    return {
        "sent": True,
        "to": to,
        "subject": subject,
        "sentAt": datetime.now().isoformat(),
        "tracking": {"delivered": True, "replied": False,
                     "materialsReceived": False},
    }


@registry.register("update_business_database", "回写业务数据库（须先经人工确认）")
def update_business_database(table: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "written": True,
        "table": table,
        "records": len(payload) if isinstance(payload, (list, dict)) else 1,
        "writtenAt": datetime.now().isoformat(),
    }


REQUIRED_TOOLS = [
    "get_ship_profile", "get_certificate_status", "generate_base_checklist",
    "get_outstanding_memos", "get_ship_risk_history",
    "search_maritime_regulations", "send_email", "update_business_database",
]
