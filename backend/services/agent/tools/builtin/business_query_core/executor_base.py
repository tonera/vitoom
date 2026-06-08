"""Shared executor primitives for business query tools."""

from __future__ import annotations

from typing import Any, Dict


def placeholder_permission_check(query_spec: Dict[str, Any], *, domain: str = "") -> Dict[str, Any]:
    return {
        "enabled": False,
        "status": "placeholder_passed",
        "note": "第一阶段保留权限校验占位；真实业务系统接入时应注入行级、字段级、操作级权限。",
        "domain": domain or query_spec.get("domain"),
        "query_spec_intent": query_spec.get("intent"),
    }


def placeholder_cost_check(query_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": False,
        "status": "placeholder_passed",
        "note": "第一阶段限制 QuerySpec schema 与 limit；后续接入查询预算和 bucket 上限。",
        "limit": query_spec.get("limit"),
    }
