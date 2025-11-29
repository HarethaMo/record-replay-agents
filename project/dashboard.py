"""Simple Streamlit dashboard for visualising benchmark logs.

Run with:

    streamlit run project/dashboard.py

This reads the JSONL log file at config.RUN_LOG_PATH and displays:
- overall accuracy by dataset and mode
- per-run table
- per-step attempt details for a selected run
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import re


from config import RUN_LOG_PATH


@dataclass
class RunRow:
    run_id: str
    dataset: str
    problem_id: str
    mode: str
    num_agents: int
    success: bool
    final_numeric_answer: float | None
    gold_answer: str
    metrics: Dict[str, Any]
    question: str


def load_runs() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    try:
        with open(RUN_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    runs.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return runs

def split_think_answer(text: str) -> tuple[Optional[str], Optional[str]]:
    """Return (think_text, answer_text) from a model output with optional tags.

    If tags are missing, returns (None, None) and caller can fall back
    to displaying the raw text.
    """
    if not text:
        return None, None

    think = None
    ans = None

    m_think = re.search(r"<think>(.*?)</think>", text, flags=re.S | re.IGNORECASE)
    if m_think:
        think = m_think.group(1).strip()

    m_ans = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.IGNORECASE)
    if m_ans:
        ans = m_ans.group(1).strip()

    return think, ans


def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent Math R&R Dashboard",
        layout="wide",
    )

    st.title("📊 Multi-Agent Math Record & Replay Dashboard")

    runs = load_runs()
    if not runs:
        st.warning("No runs found yet. Run `python -m project.benchmark` first.")
        return

    rows: List[RunRow] = []
    for r in runs:
        rows.append(
            RunRow(
                run_id=r["run_id"],
                dataset=r["dataset"],
                problem_id=r["problem_id"],
                mode=r["mode"],
                num_agents=r["num_agents"],
                success=bool(r["success"]),
                final_numeric_answer=r.get("final_numeric_answer"),
                gold_answer=r["gold_answer"],
                metrics=r.get("metrics", {}),
                question=r["question"],
            )
        )

    df = pd.DataFrame([vars(row) for row in rows])

    # Sidebar filters
    st.sidebar.header("Filters")
    dataset_sel = st.sidebar.multiselect(
        "Dataset", sorted(df["dataset"].unique()), default=list(sorted(df["dataset"].unique()))
    )
    mode_sel = st.sidebar.multiselect(
        "Mode", sorted(df["mode"].unique()), default=list(sorted(df["mode"].unique()))
    )

    df_filt = df[df["dataset"].isin(dataset_sel) & df["mode"].isin(mode_sel)]

    # Overall metrics
    st.subheader("Overall Accuracy")
    agg = (
        df_filt.groupby(["dataset", "mode"])
        .agg(
            runs=("run_id", "count"),
            success_rate=("success", "mean"),
        )
        .reset_index()
    )
    agg["success_rate"] = agg["success_rate"] * 100.0
    st.dataframe(agg.style.format({"success_rate": "{:.1f}"}), use_container_width=True)

    st.subheader("Runs")
    st.dataframe(
        df_filt[
            [
                "run_id",
                "dataset",
                "problem_id",
                "mode",
                "num_agents",
                "success",
                "final_numeric_answer",
                "gold_answer",
            ]
        ],
        use_container_width=True,
        height=300,
    )

    # Run details
    st.subheader("Run Details")
    run_ids = list(df_filt["run_id"].unique())
    if not run_ids:
        st.info("No runs match the current filter.")
        return

    selected_run_id = st.selectbox("Select run_id", run_ids)
    selected = next(r for r in runs if r["run_id"] == selected_run_id)

    st.markdown(f"**Question:** {selected['question']}")
    st.markdown(
        f"**Gold answer:** {selected['gold_answer']} &nbsp;&nbsp; "
        f"**Final numeric:** {selected.get('final_numeric_answer')} &nbsp;&nbsp; "
        f"**Mode:** `{selected['mode']}`"
    )

    # Step-by-step view
    steps = selected["steps"]
    for step in steps:
        with st.expander(f"Step {step['step_index']+1}: {step['description']}"):
            attempts = step["attempts"]
            if not attempts:
                st.write("No attempts recorded.")
                continue
            att_df = pd.DataFrame(attempts)
            st.markdown("**Attempts:**")
            st.dataframe(
                att_df[["step_index", "strategy", "model", "numeric_answer"]],
                use_container_width=True,
                height=150,
            )
            for idx, att in enumerate(attempts):
                st.markdown(f"**Attempt {idx+1} (strategy={att['strategy']}, model={att['model']}):**")
                for att_idx, att in enumerate(step["attempts"]):
                    st.markdown(
                        f"**Attempt {att_idx}** – "
                        f"strategy: `{att.get('strategy', 'unknown')}`, "
                        f"backend: `{att.get('backend', 'unknown')}`, "
                        f"model: `{att.get('model', 'unknown')}`"
                    )

                    raw_text = (
                        att.get("answer_text")
                        or att.get("reasoning")
                        or (att.get("llm_trace", {}) or {}).get("response_text", "")
                    )

                    think_text, ans_text = split_think_answer(raw_text)

                    if think_text:
                        st.markdown("Reasoning (`<think>`):")
                        st.code(think_text, language="markdown")

                    if ans_text:
                        st.markdown("Final answer (`<answer>`):")
                        st.code(ans_text, language="text")

                    if not think_text and not ans_text:
                        # Fallback for old logs / malformed outputs
                        st.markdown("Raw model output:")
                        st.code(raw_text or "[no output]", language="markdown")

                    # Numeric result from parsing
                    if "numeric_answer" in att and att["numeric_answer"] is not None:
                        st.markdown(f"Parsed numeric result: **{att['numeric_answer']}**")   

            eval_hist = step.get("evaluation_history", [])
            if eval_hist:
                st.markdown("**Evaluation history:**")
                st.dataframe(pd.DataFrame(eval_hist), use_container_width=True)


if __name__ == "__main__":  # pragma: no cover
    main()
