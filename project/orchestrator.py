"""Orchestration logic for running problems in different failure modes and interventions."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .agents import StepAgent, StepAttempt
from .config import (
    BASE_BACKEND,
    DEFAULT_NUM_AGENTS,
    LARGE_BACKEND,
    LARGE_OPENAI_BACKEND,
    MINI_OPENAI_BACKEND,
    MAX_RETRIES_PER_STEP,
    RANDOM_SEED,
    RUN_LOG_PATH,
    STOCHASTIC_FAIL_PROB,
    STOCHASTIC_DELAY_MIN,
    STOCHASTIC_DELAY_MAX,
)
from .coordinator import Coordinator
from .evaluator import StepEvaluator
from .llm_clients import LLMClient

# High-level failure mode:
Mode = Literal["stochastic", "logical"]


@dataclass
class StepLog:
    step_index: int
    description: str
    attempts: List[StepAttempt] = field(default_factory=list)
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RunLog:
    run_id: str
    timestamp: float
    dataset: str
    problem_id: str
    question: str
    gold_answer: str
    mode: Mode
    num_agents: int
    success: bool
    final_numeric_answer: Optional[float]
    final_answer_text: Optional[str]
    steps: List[StepLog]
    metrics: Dict[str, Any]


def choose_strategy_for_retry(retry_index: int) -> str:
    """Cycle through the three replay strategies in a simple way."""
    strategies = ["agent_replacement", "prompt_change", "param_change"]
    return strategies[retry_index % len(strategies)]


class Orchestrator:
    """Main engine coordinating multi-agent execution under different regimes.

    We separate:

    - `mode`:
        - "stochastic": purely stochastic failures (no evaluator).
        - "logical": evaluator-driven logical failures / replays.
    - `enable_interventions`:
        - False: no replay; first failure ends the run.
        - True: apply interventions (recalls or replacement strategies).
    - `logical_strategy` (for mode="logical" when interventions are enabled):
        - "all": cycle between all three replay strategies.
        - "agent_replacement" | "prompt_change" | "param_change":
            use only that strategy for retries.
    """

    def __init__(self) -> None:
        random.seed(RANDOM_SEED)
        self.base_client = LLMClient(BASE_BACKEND)
        self.large_client = LLMClient(LARGE_BACKEND)
        self.large_openai_client = LLMClient(LARGE_OPENAI_BACKEND)
        self.mini_openai_client = LLMClient(MINI_OPENAI_BACKEND)
        self.coord = Coordinator(self.large_openai_client)
        # We use the large backend as evaluator by default (stricter).
        self.evaluator = StepEvaluator(self.mini_openai_client)
        self.step_agent = StepAgent(self.base_client, self.large_client)

    # ------------------------------------------------------------------ core

    def run_single_problem(
        self,
        *,
        problem: Dict[str, str],
        mode: Mode,
        num_agents: int = DEFAULT_NUM_AGENTS,
        enable_interventions: bool = True,
        logical_strategy: str = "base",
    ) -> RunLog:
        """Run one problem through the multi-agent pipeline.

        Returns a RunLog with:
        - success flag
        - full step/attempt history
        - metrics including elapsed time in seconds (metrics["elapsed_sec"])
        """
        start_time = time.time()
        first_forced_failure_time: Optional[float] = None
        num_forced_stochastic_failures = 0
        num_logical_retries = 0

        question = problem["question"]
        gold_answer_str = problem["answer"]
        try:
            gold_answer = float(gold_answer_str)
        except Exception:
            gold_answer = None

        requested_num_agents = num_agents

        steps_text, coord_trace = self.coord.decompose(question, requested_num_agents)
        
        actual_num_agents = min(requested_num_agents, len(steps_text))
        steps_text = steps_text[:actual_num_agents]
        
        print(f"\n===================\nRunning Problem: {question}.\nDecomposed into {actual_num_agents} steps. \nMode: {mode}. Interventions enabled: {enable_interventions}.\n")
        
        if actual_num_agents == 0:
            # No steps to run; terminate early.
            return RunLog(
                run_id=f"{problem['id']}-{int(time.time()*1000)}",
                timestamp=time.time(),
                dataset="unknown",
                problem_id=problem["id"],
                question=question,
                gold_answer=gold_answer_str,
                mode=mode,
                num_agents=num_agents,
                success=False,
                final_numeric_answer=None,
                final_answer_text=None,
                steps=[],
                metrics={
                    "num_forced_stochastic_failures": 0,
                    "num_logical_retries": 0,
                    "terminated_early": True,
                    "terminated_reason": "no_steps_decomposed",
                    "elapsed_sec": time.time() - start_time,
                },
            )
        
        step_logs: List[StepLog] = [
            StepLog(step_index=i, description=desc) for i, desc in enumerate(steps_text)
        ]

        # Sequentially execute each step.
        for i in range(actual_num_agents):
            step_log = step_logs[i]

            # helper to early-terminate run
            def terminate_early(reason: str) -> RunLog:
                last_attempts = step_log.attempts
                final_numeric = last_attempts[-1].numeric_answer if last_attempts else None
                final_answer_text = last_attempts[-1].answer_text if last_attempts else None
                success = (
                    gold_answer is not None
                    and final_numeric is not None
                    and abs(final_numeric - gold_answer) < 1e-6
                )
                elapsed = time.time() - start_time
                return RunLog(
                    run_id=f"{problem['id']}-{int(time.time()*1000)}",
                    timestamp=time.time(),
                    dataset="unknown",
                    problem_id=problem["id"],
                    question=question,
                    gold_answer=gold_answer_str,
                    mode=mode,
                    num_agents=num_agents,
                    success=bool(success),
                    final_numeric_answer=final_numeric,
                    final_answer_text=final_answer_text,
                    steps=step_logs,
                    metrics={
                        "num_forced_stochastic_failures": num_forced_stochastic_failures,
                        "num_logical_retries": num_logical_retries,
                        "terminated_early": True,
                        "terminated_reason": reason,
                        "elapsed_sec": elapsed,
                        "time_until_first_forced_failure": first_forced_failure_time,

                    },
                )

            # ----------------------- STOCHASTIC MODE -----------------------
            if mode in ["stochastic-no-rr", "stochastic-rr"]:
                retry_count = 0
                while True:
                    # With some probability, we *pretend* this step failed.
                    if random.random() < STOCHASTIC_FAIL_PROB:
                        # add time delay to simulate LLM call
                        delay = random.uniform(STOCHASTIC_DELAY_MIN, STOCHASTIC_DELAY_MAX)
                        time.sleep(delay)
                        # run a dummy failed attempt
                        self.step_agent.run_step(
                            step_index=i,
                            problem=question,
                            step_description=step_log.description,
                            previous_steps=[
                                att for s in step_logs[:i] for att in s.attempts[-1:]],
                            strategy="base",
                        )
                        print("Forced stochastic failure on step", i)

                        num_forced_stochastic_failures += 1
                        if first_forced_failure_time is None:
                            first_forced_failure_time = time.time() - start_time
                            
                        step_log.evaluation_history.append(
                            {
                                "step_index": i,
                                "reason": "forced_stochastic_failure",
                                "retry_count": retry_count + 1,
                            }
                        )
                        if not enable_interventions:
                            # No recall allowed: run fails here.
                            return terminate_early("stochastic_failure_without_intervention")
                        # With interventions: we just try again.
                        retry_count += 1
                        if retry_count > MAX_RETRIES_PER_STEP:
                            return terminate_early("stochastic_failure_max_retries")
                        continue

                    # Actual step execution (no evaluator in stochastic mode).
                    attempt = self.step_agent.run_step(
                        step_index=i,
                        problem=question,
                        step_description=step_log.description,
                        previous_steps=[
                            att for s in step_logs[:i] for att in s.attempts[-1:]],
                        strategy="base",
                    )
                    step_log.attempts.append(attempt)
                    
                    attempts_dict = [asdict(a) for a in step_log.attempts]
                    eval_res = self.evaluator.evaluate(
                        problem=question,
                        steps=steps_text,
                        step_index=i,
                        attempts=attempts_dict,
                        final_answer=gold_answer_str,
                    )
                    step_log.evaluation_history.append(
                        {
                            "step_index": i,
                            "attempt_index": len(step_log.attempts) - 1,
                            "decision": eval_res.decision,
                            "confidence": eval_res.confidence,
                            "comment": eval_res.comment,
                            "eval_trace": {
                                "messages": eval_res.trace.messages,
                                "params": eval_res.trace.params,
                                "response_text": eval_res.trace.response_text,
                                "provider": eval_res.trace.provider,
                                "model": eval_res.trace.model,
                            },
                            "mode": "stochastic",
                        }
                    )
                    
                    if eval_res.decision == "retry":
                        return terminate_early("logical_failure_in_stochastic_mode")

                    # Otherwise accept this step and move on
                    break
            

                continue  # next step

            # ------------------------ LOGICAL MODE ------------------------
            # For logical mode we use the evaluator to decide continue vs retry.
            # Policy:
            #   - Attempt 0: base strategy
            #   - If evaluator says 'retry':
            #       * if interventions disabled -> terminate early
            #       * if interventions enabled  -> exactly ONE more attempt
            #   - After 1 retry (attempt 1) with chosen strategy:
            #       * if evaluator still says 'retry' -> terminate early
            #       * else -> accept and move to next step

            # ---- First attempt: base strategy
            attempt0 = self.step_agent.run_step(
                step_index=i,
                problem=question,
                step_description=step_log.description,
                previous_steps=[att for s in step_logs[:i] for att in s.attempts[-1:]],
                strategy="base",
            )
            step_log.attempts.append(attempt0)

            attempts_dict0 = [asdict(a) for a in step_log.attempts]
            eval_res0 = self.evaluator.evaluate(
                problem=question,
                steps=steps_text,
                step_index=i,
                attempts=attempts_dict0,
                final_answer=gold_answer_str,
            )
            step_log.evaluation_history.append(
                {
                    "step_index": i,
                    "attempt_index": len(step_log.attempts) - 1,
                    "decision": eval_res0.decision,
                    "confidence": eval_res0.confidence,
                    "comment": eval_res0.comment,
                    "eval_trace": {
                        "messages": eval_res0.trace.messages,
                        "params": eval_res0.trace.params,
                        "response_text": eval_res0.trace.response_text,
                        "provider": eval_res0.trace.provider,
                        "model": eval_res0.trace.model,
                    },
                }
            )

            if eval_res0.decision == "continue":
                # Accept base attempt and move on to next step.
                continue

            # Evaluator wants a retry after base attempt.
            if not enable_interventions:
                # Base runs: no logical retry. Run fails here.
                return terminate_early("logical_failure_without_intervention")

            # ---- Second attempt: chosen replay strategy
            num_logical_retries += 1

            # Choose strategy for this retry
            retry_strategy = logical_strategy

            attempt1 = self.step_agent.run_step(
                step_index=i,
                problem=question,
                step_description=step_log.description,
                previous_steps=[att for s in step_logs[:i] for att in s.attempts[-1:]],
                strategy=retry_strategy,
            )
            step_log.attempts.append(attempt1)

            attempts_dict1 = [asdict(a) for a in step_log.attempts]
            eval_res1 = self.evaluator.evaluate(
                problem=question,
                steps=steps_text,
                step_index=i,
                attempts=attempts_dict1,
                final_answer=gold_answer_str,
            )
            step_log.evaluation_history.append(
                {
                    "step_index": i,
                    "attempt_index": len(step_log.attempts) - 1,
                    "decision": eval_res1.decision,
                    "confidence": eval_res1.confidence,
                    "comment": eval_res1.comment,
                    "eval_trace": {
                        "messages": eval_res1.trace.messages,
                        "params": eval_res1.trace.params,
                        "response_text": eval_res1.trace.response_text,
                        "provider": eval_res1.trace.provider,
                        "model": eval_res1.trace.model,
                    },
                }
            )

            if eval_res1.decision == "continue":
                # Accept retry attempt and move on to next step.
                continue

            # Retry also failed according to evaluator; no further attempts allowed.
            return terminate_early("logical_failure_after_single_retry")


            # done with this step, go to next

        # After all steps, take the numeric result of the last step as final answer.
        last_attempts = step_logs[-1].attempts
        final_numeric = last_attempts[-1].numeric_answer if last_attempts else None
        final_answer_text = last_attempts[-1].answer_text if last_attempts else None
        success = (
            gold_answer is not None
            and final_numeric is not None
            and abs(final_numeric - gold_answer) < 1e-6
        )

        elapsed = time.time() - start_time

        return RunLog(
            run_id=f"{problem['id']}-{int(time.time()*1000)}",
            timestamp=time.time(),
            dataset="unknown",
            problem_id=problem["id"],
            question=question,
            gold_answer=gold_answer_str,
            mode=mode,
            num_agents=num_agents,
            success=bool(success),
            final_numeric_answer=final_numeric,
            final_answer_text=final_answer_text,
            steps=step_logs,
            metrics={
                "num_forced_stochastic_failures": num_forced_stochastic_failures,
                "num_logical_retries": num_logical_retries,
                "terminated_early": False,
                "terminated_reason": None,
                "elapsed_sec": elapsed,
                "time_until_first_forced_failure": first_forced_failure_time,
                "requested_num_agents": requested_num_agents,
                "actual_num_agents": actual_num_agents,
                "coordinator_trace": {
                    "messages": coord_trace.messages,
                    "params": coord_trace.params,
                    "response_text": coord_trace.response_text,
                    "provider": coord_trace.provider,
                    "model": coord_trace.model,
                },
            },
        )



def append_run_log(run: RunLog, dataset_name: str, condition: str) -> None:
    """Append a run log to the JSONL file, annotated with a high-level condition.

    `condition` is a more fine-grained label than `mode`, e.g.:

        - "base"
        - "stochastic_no_rr"
        - "stochastic_rr"
        - "logical_agent_replacement_rr"
        - "logical_prompt_change_rr"
        - "logical_param_change_rr"
    """
    data = {
        **asdict(run),
        "dataset": dataset_name,
        "condition": condition,
        "steps": [
            {
                "step_index": s.step_index,
                "description": s.description,
                "attempts": [asdict(a) for a in s.attempts],
                "evaluation_history": s.evaluation_history,
            }
            for s in run.steps
        ],
    }
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
