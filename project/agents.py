from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

from .config import (
    BASE_TEMPERATURE,
    BASE_TOP_P,
    DEEP_REASON_TEMPERATURE,
    STEP_AGENT_MAX_TOKENS,
    ALT_TEMPERATURE,
    ALT_TOP_P,
)
from .llm_clients import LLMClient


@dataclass
class LLMTrace:
    messages: List[Dict[str, str]]
    params: Dict[str, Any]
    response_text: str
    provider: str
    model: str


@dataclass
class StepAttempt:
    step_index: int
    strategy: str  # "base" | "agent_replacement" | "prompt_change" | "param_change"
    answer_text: str
    numeric_answer: Optional[float]
    backend: str
    model: str
    llm_trace: LLMTrace = field(default_factory=lambda: LLMTrace([], {}, "", "", ""))


class StepAgent:
    """Executes one step at a time using (possibly different) LLM settings."""

    def __init__(self, base_client: LLMClient, large_client: LLMClient) -> None:
        self.base_client = base_client
        self.large_client = large_client

    def _build_prompt(
        self,
        problem: str,
        step_description: str,
        previous_steps: List[StepAttempt],
        deep_reason: bool = False,
    ) -> List[Dict[str, str]]:
        history = ""
        if previous_steps:
            history_lines = []
            for s in previous_steps:
                history_lines.append(
                    f"Step {s.step_index}: result = {s.numeric_answer}, text = {s.answer_text}"
                )
            history = "Previous steps:\n" + "\n".join(history_lines) + "\n\n"

        if deep_reason:
            # Advanced agent: explicit reasoning tokens
            system_msg = (
                "You are a careful math agent. Think step-by-step inside <think>...</think>. "
                "After finishing your reasoning, output ONLY the final numeric answer "
                "inside <answer>...</answer>.\n\n"
                "FORMAT:\n"
                "<think>\n"
                "  ... your reasoning ...\n"
                "</think>\n"
                "<answer>NUMERIC_VALUE</answer>\n"
                "Do not output anything after </answer>."
            )

            user_msg = (
                f"You are solving ONLY the following step of a math problem.\n\n"
                f"Full problem:\n{problem}\n\n"
                f"{history}"
                f"Your assigned step:\n{step_description}\n\n"
                "Follow the required output format strictly. Do not explain anything "
                "outside <think>. The numeric value in <answer> must be parsable as a number."
            )

        else:
            # Base agent: no chain of thought, output only the numeric answer
            system_msg = (
                "You are a math agent. Given a math problem, you will Solve ONLY the assigned step.\n"
                "Output ONLY the final numeric result wrapped exactly like:\n"
                "<answer>NUMERIC_VALUE</answer>\n\n"
                "NO explanations. NO reasoning. NO extra text before or after the <answer> tag."
            )

            user_msg = (
                f"Full problem:\n{problem}\n\n"
                f"{history}"
                f"You are solving ONLY the following step of the math problem.\n\n"
                f"Step to solve:\n{step_description}\n\n"
                "Return ONLY:\n"
                "<answer>NUMERIC_VALUE</answer>"
            )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return messages
    
    
    def _extract_think_and_answer(self, text: str) -> tuple[Optional[str], Optional[float]]:
        """Parse <think>...</think> and <answer>NUMERIC</answer> from model output.

        Returns:
            (think_text, numeric_answer)
        """
        think = None
        numeric = None

        # Extract reasoning block if present
        m_think = re.search(r"<think>(.*?)</think>", text, flags=re.S | re.IGNORECASE)
        if m_think:
            think = m_think.group(1).strip()

        # Extract numeric answer
        m_ans = re.search(
            r"<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>",
            text,
            flags=re.S | re.IGNORECASE,
        )
        if m_ans:
            try:
                numeric = float(m_ans.group(1).strip())
            except Exception:
                numeric = None

        return think, numeric



    def run_step(
        self,
        *,
        step_index: int,
        problem: str,
        step_description: str,
        previous_steps: List[StepAttempt],
        strategy: str = "base",
    ) -> StepAttempt:
        """Execute one step according to the chosen strategy."""
        if strategy == "agent_replacement":
            client = self.large_client
            temperature = BASE_TEMPERATURE
            top_p = BASE_TOP_P
            deep_reason = True
        elif strategy == "prompt_change":
            client = self.base_client
            temperature = DEEP_REASON_TEMPERATURE
            top_p = BASE_TOP_P
            deep_reason = True
        elif strategy == "param_change":
            client = self.base_client
            temperature = ALT_TEMPERATURE
            top_p = ALT_TOP_P
            deep_reason = False
        else:  # "base"
            client = self.base_client
            temperature = BASE_TEMPERATURE
            top_p = BASE_TOP_P
            deep_reason = False

        messages = self._build_prompt(
            problem=problem,
            step_description=step_description,
            previous_steps=previous_steps,
            deep_reason=deep_reason,
        )

        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": STEP_AGENT_MAX_TOKENS,
        }

        resp = client.chat(
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            top_p=params["top_p"],
        )
        text = resp.content.strip()

        print(f"Step Agent Response (step {step_index}, strategy={strategy}):", text)
        # New: parse <think> and <answer>
        _think, numeric = self._extract_think_and_answer(text)

        trace = LLMTrace(
            messages=messages,
            params=params,
            response_text=text,
            provider=client.cfg.provider,
            model=client.cfg.model,
        )

        return StepAttempt(
            step_index=step_index,
            strategy=strategy,
            answer_text=text,           # full raw output (<think> + <answer>)
            numeric_answer=numeric,     # strict numeric parsing from <answer>
            backend=client.cfg.provider,
            model=client.cfg.model,
            llm_trace=trace,
        )

