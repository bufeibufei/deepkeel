from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ReferenceProjection:
    references: list[dict[str, Any]]
    evidence: list[dict[str, Any]]


class ReferenceProjector(Protocol):
    def __call__(
        self,
        tool_results: list[dict[str, Any]],
        final_answer: dict[str, Any],
    ) -> ReferenceProjection: ...


class DefaultReferenceProjector:
    """Collect generic source records without knowing product tool names."""

    def __init__(self, *, limit: int = 12) -> None:
        self.limit = max(1, int(limit))

    def __call__(
        self,
        tool_results: list[dict[str, Any]],
        final_answer: dict[str, Any],
    ) -> ReferenceProjection:
        candidates: list[dict[str, Any]] = []
        for item in final_answer.get("references") or []:
            if isinstance(item, dict):
                candidates.append(_normalize_reference(item, source_tool="agent.final"))
        for tool_result in tool_results:
            tool_name = str(tool_result.get("name") or "")
            data = tool_result.get("data") if isinstance(tool_result.get("data"), dict) else {}
            query = str(data.get("query") or "")
            for reference, is_evidence in _nested_reference_candidates(data):
                candidates.append(
                    _normalize_reference(
                        reference,
                        source_tool=tool_name,
                        query=query,
                        is_evidence=is_evidence,
                    )
                )
            for artifact in tool_result.get("artifacts") or []:
                if not isinstance(artifact, dict):
                    continue
                artifact_data = artifact.get("data") if isinstance(artifact.get("data"), dict) else {}
                for reference, is_evidence in _nested_reference_candidates(artifact_data):
                    candidates.append(
                        _normalize_reference(
                            reference,
                            source_tool=tool_name,
                            query=query,
                            is_evidence=is_evidence,
                        )
                    )

        references: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate:
                continue
            identity = str(
                candidate.get("unit_id")
                or candidate.get("url")
                or candidate.get("reference_id")
                or f"{candidate.get('kind')}:{candidate.get('title')}:{candidate.get('snippet')}"
            )
            if identity in seen:
                continue
            seen.add(identity)
            references.append(candidate)
            if len(references) >= self.limit:
                break
        evidence = [item for item in references if bool(item.get("is_evidence"))]
        return ReferenceProjection(references=references, evidence=evidence)


def _nested_reference_candidates(
    value: Any,
    *,
    depth: int = 0,
    evidence: bool = False,
) -> list[tuple[dict[str, Any], bool]]:
    if depth > 5:
        return []
    if isinstance(value, list):
        candidates: list[tuple[dict[str, Any], bool]] = []
        for item in value:
            candidates.extend(
                _nested_reference_candidates(item, depth=depth + 1, evidence=evidence)
            )
        return candidates
    if not isinstance(value, dict):
        return []
    candidates = []
    reference_keys = {
        "citations",
        "evidence",
        "evidence_refs",
        "evidence_ledger",
        "references",
        "results",
    }
    for key, item in value.items():
        child_evidence = evidence or key in {"evidence", "evidence_refs", "evidence_ledger"}
        if key in reference_keys and isinstance(item, list):
            candidates.extend(
                (reference, child_evidence)
                for reference in item
                if isinstance(reference, dict)
            )
            continue
        if isinstance(item, (dict, list)):
            candidates.extend(
                _nested_reference_candidates(
                    item,
                    depth=depth + 1,
                    evidence=child_evidence,
                )
            )
    return candidates


def _normalize_reference(
    value: dict[str, Any],
    *,
    source_tool: str = "",
    query: str = "",
    is_evidence: bool = False,
) -> dict[str, Any]:
    unit_id = str(value.get("unit_id") or value.get("id") or "").strip()
    url = str(value.get("url") or "").strip()
    resolved_kind = str(value.get("kind") or ("web" if url else "record" if unit_id else ""))
    title = str(value.get("title") or value.get("title_cn") or value.get("site_name") or "").strip()
    snippet = str(
        value.get("snippet")
        or value.get("summary")
        or value.get("text_preview")
        or ""
    ).strip()[:360]
    if not resolved_kind or (not unit_id and not url and not title):
        return {}
    reference_id = unit_id or url or f"{resolved_kind}:{title}:{snippet[:80]}"
    normalized = {
        "reference_id": reference_id,
        "kind": resolved_kind,
        "title": title or ("Web source" if resolved_kind == "web" else "Reference"),
        "snippet": snippet,
        "source_tool": source_tool or str(value.get("source_tool") or ""),
        "query": query or str(value.get("query") or ""),
        "is_evidence": bool(value.get("is_evidence") or is_evidence),
    }
    optional = {
        "unit_id": unit_id,
        "url": url,
        "site_name": str(value.get("site_name") or ""),
        "publish_time": str(value.get("publish_time") or ""),
        "flow": str(value.get("flow") or ""),
        "quality_grade": str(value.get("quality_grade") or value.get("grade") or ""),
        "page": str(value.get("page") or ""),
        "chapter": str(value.get("chapter") or ""),
        "source_file": str(value.get("source_file") or ""),
    }
    normalized.update({key: item for key, item in optional.items() if item})
    return normalized
