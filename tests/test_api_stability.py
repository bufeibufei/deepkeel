from __future__ import annotations

from deepkeel.api_stability import _annotation_name


class _RenderedAnnotation:
    def __init__(self, rendered: str) -> None:
        self.rendered = rendered

    def __str__(self) -> str:
        return self.rendered


def test_annotation_name_normalizes_optional_rendering_across_python_versions() -> None:
    legacy = _annotation_name(_RenderedAnnotation("typing.Optional[Literal['fast']]"))
    pep604 = _annotation_name(_RenderedAnnotation("Literal['fast'] | None"))

    assert legacy == pep604 == "Optional[Literal['fast']]"


def test_annotation_name_ignores_forward_ref_interpreter_metadata() -> None:
    legacy = _annotation_name(_RenderedAnnotation("ForwardRef('list[ToolCall]')"))
    python314 = _annotation_name(
        _RenderedAnnotation("ForwardRef('list[ToolCall]', is_class=True)")
    )

    assert legacy == python314 == "ForwardRef('list[ToolCall]')"
