"""Benchmark runner for the multi-agent math system.

It:
- loads small GSM8K-style samples,
- runs them under several conditions:

    * base                     (no interventions)
    * stochastic_no_rr         (stochastic failures, no recall)
    * stochastic_rr            (stochastic failures, recall base agent)
    * logical_agent_replacement_rr
    * logical_prompt_change_rr
    * logical_param_change_rr

- logs all runs to JSONL,
- prints a summary table with Accuracy and Time per dataset & condition,
- prints deltas vs the base condition.

Usage:

    python -m project.benchmark

Optional CLI args:
    --suite all|stochastic|logical|base
    --n N_AGENTS
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm  # progress bar
import os

from .config import DEFAULT_NUM_AGENTS, RUN_LOG_PATH
from .datasets import GSM8K_HARD_SAMPLES, GSM8K_SAMPLES
from dataclasses import asdict
from .orchestrator import Orchestrator, append_run_log

def warmup(orch: Orchestrator, num_agents: int) -> None:
    """Run a small untimed pass to trigger model loading & CUDA warmup.

    This is NOT logged and NOT included in metrics.
    """
    if not GSM8K_SAMPLES:
        return

    dummy = GSM8K_SAMPLES[0]
    q = dummy["question"]
    # 1) Coordinator warmup
    steps_text, coord_trace = orch.coord.decompose(q, num_agents)

    # 2) One base step + one "large" step
    #    (so both HF models get touched)
    base_attempt = orch.step_agent.run_step(
        step_index=0,
        problem=q,
        step_description=steps_text[0],
        previous_steps=[],
        strategy="base",
    )
    large_attempt = orch.step_agent.run_step(
        step_index=0,
        problem=q,
        step_description=steps_text[0],
        previous_steps=[base_attempt],
        strategy="agent_replacement",  # uses the "large" client
    )

    # 3) Evaluator warmup
    attempts_dict = [asdict(base_attempt), asdict(large_attempt)]
    orch.evaluator.evaluate(
        problem=q,
        steps=steps_text,
        step_index=0,
        attempts=attempts_dict,
        final_answer=dummy["answer"],
    )

    # nothing returned, nothing logged

def run_benchmark(
    suite: str = "all",
    num_agents: int = DEFAULT_NUM_AGENTS,
) -> None:
    orch = Orchestrator()

    print("Warming up models (coordinator, agents, evaluator)...")
    warmup(orch, num_agents)
    print("Warmup done. Starting benchmark.\n")
    
    datasets = [
        ("gsm8k", GSM8K_SAMPLES),
        # ("gsm8k_hard", GSM8K_HARD_SAMPLES),
    ]

    # Define experimental conditions with their orchestrator configs.
    all_conditions = {
        # Support base runs without any intervention.
        "base": dict(mode="logical", enable_interventions=False, logical_strategy="all"),
        # Stochastic failures: compare no interventions vs recall (re-run the step).
        "stochastic_no_rr": dict(mode="stochastic", enable_interventions=False),
        "stochastic_rr": dict(mode="stochastic", enable_interventions=True),
        # Logical failures: compare base vs each replacement method.
        "logical_agent_replacement_rr": dict(
            mode="logical", enable_interventions=True, logical_strategy="agent_replacement"
        ),
        "logical_prompt_change_rr": dict(
            mode="logical", enable_interventions=True, logical_strategy="prompt_change"
        ),
        # "logical_param_change_rr": dict(
        #     mode="logical", enable_interventions=True, logical_strategy="param_change"
        # ),
    }

    if suite == "base":
        selected_conditions = ["base"]
    elif suite == "stochastic":
        selected_conditions = ["base", "stochastic_no_rr", "stochastic_rr"]
    elif suite == "logical":
        selected_conditions = [
            "base",
            "logical_agent_replacement_rr",
            "logical_prompt_change_rr",
            # "logical_param_change_rr",
        ]
    else:  # "all"
        selected_conditions = list(all_conditions.keys())

    # results[dataset][condition] -> list of dicts {success: bool, time: float}
    results: Dict[str, Dict[str, List[Dict[str, float | bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    total_runs = sum(len(problems) for _, problems in datasets) * len(selected_conditions)

    with tqdm(total=total_runs, desc="Running benchmark") as pbar:
        for dataset_name, problems in datasets:
            for problem in problems:
                for cond_name in selected_conditions:
                    cfg = all_conditions[cond_name]
                    run = orch.run_single_problem(
                        problem=problem,
                        num_agents=num_agents,
                        **cfg,  # mode, enable_interventions, logical_strategy?
                    )
                    append_run_log(run, dataset_name, cond_name)

                    elapsed = float(run.metrics.get("elapsed_sec", 0.0))
                    results[dataset_name][cond_name].append(
                        {
                            "success": bool(run.success),
                            "time": elapsed,
                        }
                    )

                    pbar.update(1)

    # Print summary table
    print("\n=== Multi-Agent Math Benchmark Results ===")
    print(f"Num agents (steps): {num_agents}")
    header = f"{'Dataset':<12} {'Condition':<28} {'Runs':>5} {'Accuracy%':>10} {'AvgTime(s)':>12}"
    print(header)
    print("-" * len(header))

    summary_per_dataset: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    for dataset_name, conds in results.items():
        for cond_name, records in conds.items():
            n = len(records)
            if n == 0:
                continue
            acc = sum(1 for r in records if r["success"]) / n
            avg_time = sum(float(r["time"]) for r in records) / n
            summary_per_dataset[dataset_name][cond_name] = {
                "acc": acc,
                "time": avg_time,
                "runs": n,
            }
            print(
                f"{dataset_name:<12} {cond_name:<28} "
                f"{n:>5} {acc*100:>9.1f}% {avg_time:>11.2f}"
            )

    # Overall summary across all datasets
    all_records: Dict[str, List[Dict[str, float | bool]]] = defaultdict(list)
    for _, modes in results.items():
        for cond_name, records in modes.items():
            all_records[cond_name].extend(records)

    print("\nOverall (all datasets combined):")
    header2 = f"{'Condition':<28} {'Runs':>5} {'Accuracy%':>10} {'AvgTime(s)':>12}"
    print(header2)
    print("-" * len(header2))

    overall_summary: Dict[str, Dict[str, float]] = {}

    for cond_name, records in all_records.items():
        n = len(records)
        if n == 0:
            continue
        acc = sum(1 for r in records if r["success"]) / n
        avg_time = sum(float(r["time"]) for r in records) / n
        overall_summary[cond_name] = {"acc": acc, "time": avg_time, "runs": n}
        print(
            f"{cond_name:<28} {n:>5} {acc*100:>9.1f}% {avg_time:>11.2f}"
        )

    # Compare each intervention condition to the base condition (overall).
    base_overall = overall_summary.get("base")
    if base_overall:
        print("\nDeltas vs base (overall):")
        header3 = f"{'Condition':<28} {'ΔAcc(ppts)':>12} {'ΔTime(s)':>12}"
        print(header3)
        print("-" * len(header3))
        base_acc = base_overall["acc"]
        base_time = base_overall["time"]
        for cond_name, stats in overall_summary.items():
            if cond_name == "base":
                continue
            d_acc = (stats["acc"] - base_acc) * 100.0  # percentage points
            d_time = stats["time"] - base_time
            print(
                f"{cond_name:<28} {d_acc:>11.2f} {d_time:>11.2f}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent math benchmark")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "stochastic", "logical", "base"],
        help="Which subset of conditions to run",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_NUM_AGENTS,
        help="Number of agents / steps",
    )
    parser.add_argument(
        "--reset_logs",
        action="store_true",
        help="If set, clears existing log file before running benchmark",
    )
    args = parser.parse_args()
    reset_logs = args.reset_logs
    if reset_logs:
        if os.path.exists(RUN_LOG_PATH):
            os.remove(RUN_LOG_PATH)

    run_benchmark(suite=args.suite, num_agents=args.n)


if __name__ == "__main__":  # pragma: no cover
    main()
