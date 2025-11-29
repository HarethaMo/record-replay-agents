from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import json

from .config import EVALUATOR_MAX_TOKENS
from .llm_clients import LLMClient


@dataclass
class EvalTrace:
    messages: List[Dict[str, str]]
    params: Dict[str, Any]
    response_text: str
    provider: str
    model: str


@dataclass
class EvalResult:
    decision: str  # "continue" or "retry"
    confidence: float
    comment: str
    trace: EvalTrace = field(default_factory=lambda: EvalTrace([], {}, "", "", ""))


class StepEvaluator:
    """LLM-based evaluator that decides whether to accept a step or retry."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def evaluate(
        self,
        *,
        problem: str,
        steps: List[str],
        step_index: int,
        attempts: List[Dict[str, Any]],
        final_answer: str,
    ) -> EvalResult:
        steps_str = "\n".join(f"{i}. {s}" for i, s in enumerate(steps))
        attempts_str = json.dumps(attempts, indent=2, ensure_ascii=False)
        

        system_msg = (
            "You are a strict math step evaluator. "
            "Given the full problem, the plan steps, and the attempts made for the current step, "
            "you decide whether the latest attempt is acceptable or the step must be retried.\n\n"
            "Respond in strict JSON with keys: decision, confidence, comment.\n"
            "decision must be either 'continue' or 'retry'."
            "The comment must be short and precise.\n"
            "Only output the JSON, nothing else."
        )



        user_msg = (
            f"Problem:\n{problem}\n\n"
            f"Overall step plan:\n{steps_str}\n\n"
            f"Current step index: {step_index}\n\n"
            f"Attempts for this step (JSON):\n{attempts_str}\n\n"
            f"Gold final answer (for reference, if known): {final_answer}\n"
            "Response JSON format:\n"
            '{ "decision": "continue" | "retry", "confidence": float (0.0-1.0), "comment": "explanation text" }\n'
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": EVALUATOR_MAX_TOKENS,
        }

        resp = self.client.chat(
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            top_p=params["top_p"],
        )
        
        print("Evaluator Response:", resp.content)

        raw_text = resp.content.strip()

        # Default fallback
        decision = "continue"
        confidence = 0.5
        comment = "Default accept (fallback)."

        try:
            data = json.loads(raw_text)
            decision = str(data.get("decision", decision))
            confidence = float(data.get("confidence", confidence))
            comment = str(data.get("comment", comment))
            if decision not in ("continue", "retry"):
                decision = "continue"
        except Exception:
            comment = f"Failed to parse evaluator JSON, raw: {raw_text[:200]}"

        trace = EvalTrace(
            messages=messages,
            params=params,
            response_text=raw_text,
            provider=self.client.cfg.provider,
            model=self.client.cfg.model,
        )

        return EvalResult(
            decision=decision,
            confidence=confidence,
            comment=comment,
            trace=trace,
        )
