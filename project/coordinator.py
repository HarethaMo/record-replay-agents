from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .config import COORDINATOR_MAX_TOKENS
from .llm_clients import LLMClient


@dataclass
class CoordinatorTrace:
    messages: List[Dict[str, str]]
    params: Dict[str, Any]
    response_text: str
    provider: str
    model: str


class Coordinator:
    """LLM-based coordinator that decomposes a problem into N steps."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def decompose(self, problem: str, num_steps: int) -> Tuple[List[str], CoordinatorTrace]:
        """Ask the LLM to produce exactly `num_steps` step descriptions.

        Returns
        -------
        steps: list of step description strings
        trace: CoordinatorTrace with full prompt/response info
        """
        system_msg = (
            "You are a math task decomposition assistant. "
            "Given a math word problem and a desired number of steps, "
            "you must produce AT MOST that many logically ordered sub-steps. "
            "using only the previous step's result and the original problem."
            "Each step should be a short instruction that can be executed locally."
            "A single step must at least have one mathematical operation, but it can be more complex."
            "Each step should only include instructions, IT MUST NOT include any calculations, solutions, or answers."
        )

        user_msg = (
            f"Problem:\n{problem}\n\n"
            f"Decompose this problem into AT MOST {num_steps} numbered steps. "
            "A step can have one or more mathematical operations.\n"
            "Return them as a list, one per line, in the form:\n"
            "1. ...\n2. ...\n...\n"
            "Do NOT include any extra text or explanation. Output ONLY the steps.\n"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        params = {
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": COORDINATOR_MAX_TOKENS,
        }

        resp = self.client.chat(
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            top_p=params["top_p"],
        )
        print("Coordinator Response:", resp.content)

        text = resp.content.strip()
        # Very simple parsing: look for lines starting with digits + dot.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        steps: List[str] = []
        for ln in lines:
            if ln[0].isdigit():
                # remove leading "1." / "2)" etc.
                parts = ln.split(".", 1)
                if len(parts) == 2:
                    steps.append(parts[1].strip())
                else:
                    steps.append(ln)
            else:
                steps.append(ln)

        # Ensure we have exactly num_steps (truncate or pad if necessary)
        if len(steps) > num_steps:
            steps = steps[:num_steps]
        elif len(steps) < num_steps:
            while len(steps) < num_steps:
                steps.append(f"Fallback step {len(steps)+1}: compute any missing values.")

        trace = CoordinatorTrace(
            messages=messages,
            params=params,
            response_text=text,
            provider=self.client.cfg.provider,
            model=self.client.cfg.model,
        )

        return steps, trace
