from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from deepkeel.contracts import ToolCall


@dataclass(frozen=True, slots=True)
class EntryToolActivationRequest:
    """Context available when a model selects a Skill package entry tool."""

    tool_calls: tuple[ToolCall, ...]
    current_activation: Mapping[str, Any] = field(default_factory=dict)
    run_id: str = ""
    user_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    question: str = ""
    messages: tuple[Mapping[str, Any], ...] = ()
    context_bundle: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EntryToolActivationDecision:
    """A trusted Skill snapshot plus optional normalized tool calls."""

    skill_activation: Mapping[str, Any]
    tool_calls: tuple[ToolCall, ...] = ()
    reason: str = "entry_tool_selected"


class EntryToolSkillActivator(Protocol):
    """Host-owned policy for promoting an entry-tool call into a Skill."""

    def activate(
        self,
        request: EntryToolActivationRequest,
    ) -> EntryToolActivationDecision | None: ...
