"""Multi-agent math system with record & replay style recovery.

This package implements:
- A coordinator LLM that decomposes math problems into N steps.
- N step-agents that each solve *only one* step, passing their result forward.
- An evaluator that decides whether to accept a step or trigger replay.
- Two modes of robustness testing:
    1) Stochastic failure and retry.
    2) Logical failure with three replay strategies:
       - agent replacement
       - prompt change
       - parameter change

The project supports both OpenAI models and local LLMs exposed via an
OpenAI-compatible API such as Ollama.
"""
