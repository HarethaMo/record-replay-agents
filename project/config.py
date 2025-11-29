"""Global configuration for the multi-agent math record & replay project."""

import os
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

BackendProvider = Literal["openai", "transformers"]


@dataclass
class LLMBackendConfig:
    """Configuration for a single LLM backend.

    Parameters
    ----------
    provider:
        - "openai": uses the OpenAI Python client.
        - "transformers": uses a local HuggingFace model via transformers.
    base_url:
        For "openai": usually "https://api.openai.com/v1".
        For "transformers": ignored.
    api_key:
        For "openai": your OpenAI API key.
        For "transformers": ignored.
    model:
        - For "openai": the model name (e.g. "gpt-4o-mini").
        - For "transformers": the HF model id or local path
          (e.g. "Qwen/Qwen2.5-1.5B-Instruct" or "./local-model-dir").
    """

    provider: BackendProvider
    base_url: str
    api_key: str
    model: str


# Default: local transformers model as the *base* agent,
# and OpenAI as the *larger* / more reliable replacement agent.

BASE_BACKEND = LLMBackendConfig(
    provider=os.getenv("BASE_PROVIDER", "transformers"),  # "openai" or "transformers"
    base_url=os.getenv("BASE_LLM_BASE_URL", ""),          # unused for transformers
    api_key=os.getenv("BASE_LLM_API_KEY", ""),            # unused for transformers
    model=os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
)

LARGE_BACKEND = LLMBackendConfig(
    provider=os.getenv("BASE_PROVIDER", "transformers"),  # "openai" or "transformers"
    base_url=os.getenv("BASE_LLM_BASE_URL", ""),          # unused for transformers
    api_key=os.getenv("BASE_LLM_API_KEY", ""),            # unused for transformers
    model=os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
)

OpenAI_BACKEND = LLMBackendConfig(
    provider=os.getenv("OpenAI_PROVIDER", "openai"),
    base_url=os.getenv("OpenAI_LLM_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OpenAI_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY")),
    model=os.getenv("OpenAI_MODEL_NAME", "gpt-5"),
)


# ---------------------------------------------------------------------------
# Experiment knobs
# ---------------------------------------------------------------------------

# Number of agents / steps to use in the decomposition.
DEFAULT_NUM_AGENTS: int = int(os.getenv("NUM_AGENTS", "3"))

# Probability of forcing a stochastic failure in mode="stochastic".
STOCHASTIC_FAIL_PROB: float = float(os.getenv("STOCHASTIC_FAIL_PROB", "0.2"))

# Max retries per step (for logical failures and stochastic forced retries).
MAX_RETRIES_PER_STEP: int = int(os.getenv("MAX_RETRIES_PER_STEP", "3"))

# Maximum tokens for each LLM call in different roles.
COORDINATOR_MAX_TOKENS: int = int(os.getenv("COORDINATOR_MAX_TOKENS", "512"))
STEP_AGENT_MAX_TOKENS: int = int(os.getenv("STEP_AGENT_MAX_TOKENS", "512"))
EVALUATOR_MAX_TOKENS: int = int(os.getenv("EVALUATOR_MAX_TOKENS", "512"))

# Base parameters for the step agents.
BASE_TEMPERATURE: float = float(os.getenv("BASE_TEMPERATURE", "0.1"))
BASE_TOP_P: float = float(os.getenv("BASE_TOP_P", "1.0"))

# Parameters for the "prompt change" strategy (think more).
DEEP_REASON_TEMPERATURE: float = float(os.getenv("DEEP_REASON_TEMPERATURE", "0.2"))

# Parameters for the "parameter change" strategy.
ALT_TEMPERATURE: float = float(os.getenv("ALT_TEMPERATURE", "0.4"))
ALT_TOP_P: float = float(os.getenv("ALT_TOP_P", "0.9"))

# Random seed for reproducibility in benchmarks.
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# Paths
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

RUN_LOG_PATH: str = os.path.join(LOG_DIR, "runs.jsonl")
