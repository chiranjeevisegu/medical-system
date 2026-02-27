from __future__ import annotations

from typing import Any, Callable

try:
    from langchain_core.runnables import RunnableLambda
except Exception:  # pragma: no cover - fallback when langchain is unavailable
    RunnableLambda = None  # type: ignore[assignment]


State = dict[str, Any]
StepFn = Callable[[State], State]


class SequentialAgentFlow:
    """Sequential state-passing flow, backed by LangChain when available."""

    def __init__(self, steps: list[StepFn]) -> None:
        self.steps = steps
        self._chain = self._build_chain(steps)

    def run(self, state: State) -> State:
        if self._chain is not None:
            return self._chain.invoke(state)

        current = state
        for step in self.steps:
            current = step(current)
        return current

    @staticmethod
    def _build_chain(steps: list[StepFn]):
        if RunnableLambda is None:
            return None
        if not steps:
            return RunnableLambda(lambda x: x)

        chain = RunnableLambda(steps[0])
        for step in steps[1:]:
            chain = chain | RunnableLambda(step)
        return chain