from __future__ import annotations

from deepkeel.runtime_sdk import HarnessRuntimeBuilder, RuntimeRequest


class LocalProvider:
    """Small deterministic provider used to keep the quickstart offline."""

    model = "quickstart-model"
    model_role = "fast"

    def complete_chat(self, messages, **_kwargs):
        question = str(messages[-1].get("content") or "")
        return {
            "message": {
                "role": "assistant",
                "content": f"DeepKeel received: {question}",
            },
            "finish_reason": "stop",
            "model": self.model,
        }


def run_quickstart() -> str:
    runtime = HarnessRuntimeBuilder().build()
    result = runtime.run(
        RuntimeRequest(
            question="Is the runtime ready?",
            user_id="quickstart-user",
            context_bundle={"agent_session_id": "quickstart-run"},
        ),
        provider=LocalProvider(),
    )
    return result.final_answer.markdown


if __name__ == "__main__":
    print(run_quickstart())
