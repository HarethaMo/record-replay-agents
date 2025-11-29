"""Utility wrappers around OpenAI-compatible and local transformers LLMs.

We intentionally keep this file small and framework-free so students can see
the raw usage.

Backends:

- OpenAI (via `openai` Python package)
- Local HuggingFace transformers models (via `transformers` + `torch`)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from .config import LLMBackendConfig

@dataclass
class ChatResult:
    content: str
    raw: Any


class LLMClient:
    """Simple chat client for either OpenAI or local transformers models."""

    def __init__(self, cfg: LLMBackendConfig) -> None:
        self.cfg = cfg
        self._client = None  # OpenAI client if used

        # transformers-specific fields
        self._tf_model = None
        self._tf_tokenizer = None
        self._tf_device: Optional[str] = None

        if cfg.provider == "openai":
            if OpenAI is None:
                raise RuntimeError(
                    "openai package not installed. Run `pip install openai`."
                )
                
            self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

        elif cfg.provider == "transformers":
            # Lazy import so that users who only use OpenAI don't need transformers.
            try:
                import torch  # type: ignore
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "To use provider='transformers', install transformers and torch:\n"
                    "  pip install transformers torch\n"
                ) from e

            self._tf_device = "cuda"

            # You can point cfg.model to a HF hub id or local path.
            self._tf_tokenizer = AutoTokenizer.from_pretrained(cfg.model)
            self._tf_model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16)
            self._tf_model.to(self._tf_device)
            self._tf_model.eval()

        else:
            raise ValueError(f"Unsupported provider: {cfg.provider}")

    # ------------------------------------------------------------------ utils

    def _build_transformers_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat-style messages into a single prompt string.

        If the tokenizer supports `apply_chat_template`, we use it.
        Otherwise we fall back to a simple 'ROLE: content' transcript.
        """
        assert self._tf_tokenizer is not None
        tok = self._tf_tokenizer

        # Try native chat template if available (works for many chat models).
        if getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback: naive role-tagged transcript
        parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    # ------------------------------------------------------------------ main API

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> ChatResult:
        """Send a chat completion request and return the text content."""

        if self.cfg.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                # temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
            )
            msg = resp.choices[0].message
            return ChatResult(content=msg.content or "", raw=resp)

        if self.cfg.provider == "transformers":
            assert self._tf_model is not None and self._tf_tokenizer is not None
            import torch  # type: ignore

            prompt = self._build_transformers_prompt(messages)
            enc = self._tf_tokenizer(
                prompt,
                return_tensors="pt",
            )
            enc = {k: v.to(self._tf_device) for k, v in enc.items()}

            # Greedy vs sampled generation depending on temperature.
            do_sample = temperature > 0.0

            with torch.no_grad():
                out = self._tf_model.generate(
                    **enc,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    max_new_tokens=max_tokens,
                    pad_token_id=self._tf_tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens.
            gen_ids = out[0][enc["input_ids"].shape[-1] :]
            text = self._tf_tokenizer.decode(gen_ids, skip_special_tokens=True)
            return ChatResult(content=text, raw={"generated_text": text})

        # Should never get here because of __init__ checks.
        raise RuntimeError(f"Unsupported provider at runtime: {self.cfg.provider}")
