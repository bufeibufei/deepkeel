from __future__ import annotations

import ast
import json
from typing import Any

from harness_core.subagents.contracts import SubAgentSpec
from harness_core.type_narrowing import as_dict

def _json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _fallback_subagent_output(value: str) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    if not text or text in {"{}", "[]", "null"}:
        return None
    if text.startswith("{"):
        for key in ("conclusion", "summary"):
            marker = f'"{key}"'
            marker_index = text.find(marker)
            if marker_index < 0:
                continue
            value_start = text.find(":", marker_index + len(marker))
            quote_start = text.find('"', value_start + 1)
            quote_end = text.find('"', quote_start + 1)
            if quote_start >= 0 and quote_end > quote_start:
                candidate = text[quote_start + 1 : quote_end].strip()
                if candidate:
                    text = candidate
                    break
    return {
        "conclusion": text[:4000],
        "evidence": [],
        "evidence_refs": [],
        "risks": [],
        "recommendations": [],
        "claims": [],
        "warnings": ["The model returned usable text but failed schema validation; the lead agent should lower confidence and verify."],
        "confidence": 0.35,
        "abstained": False,
    }


def _validated_json(value: str, schema: dict[str, Any]) -> dict[str, Any]:
    parsed = _json_object(value)
    _validate_output(parsed, schema)
    return parsed


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = _readable_list_item(item)
        if text:
            result.append(text)
    return result


def _readable_list_item(value: Any) -> str:
    if isinstance(value, dict):
        return _dict_summary(value)
    text = str(value or "").strip()
    if not text or not (text.startswith("{") and text.endswith("}")):
        return text
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
    return _dict_summary(parsed) if isinstance(parsed, dict) else text


def _dict_summary(value: dict[str, Any]) -> str:
    for key in ("summary", "conclusion", "claim", "text", "description", "title", "fact"):
        text = str(value.get(key) or "").strip()
        if text:
            return text
    return "; ".join(
        f"{key}: {item}"
        for key, item in list(value.items())[:4]
        if item not in (None, "", [], {})
    )


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return None


def _output_schema(spec: SubAgentSpec) -> dict[str, Any]:
    default_properties: dict[str, Any] = {
        "conclusion": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": {"type": "object"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "claims": {"type": "array", "items": {"type": "object"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": ["number", "null"]},
        "abstained": {"type": "boolean"},
    }
    contract = dict(spec.output_contract or {})
    nested = contract.get("schema")
    if isinstance(nested, dict):
        return dict(nested)
    if contract.get("type") == "object" and isinstance(contract.get("properties"), dict):
        schema = dict(contract)
        schema.setdefault("additionalProperties", False)
        return schema
    properties = dict(default_properties)
    properties.update(contract.get("properties") or {})
    required = contract.get("required")
    if not isinstance(required, list) or not required:
        required = ["conclusion", "evidence", "risks", "recommendations"]
    for key in required:
        properties.setdefault(str(key), {})
    return {
        "type": "object",
        "properties": properties,
        "required": [str(key) for key in required],
        "additionalProperties": bool(contract.get("additionalProperties", False)),
    }


def _validate_output(value: dict[str, Any], schema: dict[str, Any]) -> None:
    if not value:
        raise RuntimeError("subagent returned invalid JSON")
    missing = [str(key) for key in schema.get("required") or [] if key not in value]
    if missing:
        raise RuntimeError(f"subagent output is missing required fields: {', '.join(missing)}")
    properties = as_dict(schema.get("properties"))
    for key, rule in properties.items():
        if key not in value or not isinstance(rule, dict) or not rule.get("type"):
            continue
        expected = rule["type"]
        valid = (
            any(_matches_json_type(value[key], str(item)) for item in expected)
            if isinstance(expected, list)
            else _matches_json_type(value[key], str(expected))
        )
        if not valid:
            raise RuntimeError(f"subagent output field has invalid type: {key}")


def _validate_input(value: dict[str, Any], contract: dict[str, Any]) -> None:
    required = contract.get("required") if isinstance(contract, dict) else []
    if not isinstance(required, list):
        return
    missing = [
        str(key)
        for key in required
        if str(key) not in value or value.get(str(key)) in (None, "", [], {})
    ]
    if missing:
        raise RuntimeError(f"subagent input is missing required fields: {', '.join(missing)}")


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True
