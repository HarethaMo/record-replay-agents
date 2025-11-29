# Multi-Agent Math Reasoning Benchmark with Record-and-Replay (R&R)

This project implements a multi-agent, step-decomposed math reasoning system with stochastic failures, logical failures, and single-step recovery interventions. It supports both local Transformer models and OpenAI models, and logs full traces for analysis through a built-in dashboard.

---

## Overview

The system processes a math word problem using three components:

### 1. Coordinator (Planning Agent)
Breaks a problem into *N* reasoning steps.

### 2. Step Agents
Each step is solved independently by an LLM.  
Two modes:
- **Base agent**: outputs only `<answer>NUMBER</answer>`
- **Advanced agent**: outputs `<think>…</think><answer>NUMBER</answer>`

### 3. Evaluator
Checks each step’s correctness and decides whether to:
- continue, or  
- retry once using a chosen intervention strategy.

All calls—coordinator, agents, evaluator—are logged per question.

---

## Failure Modes

### Stochastic Failure
A synthetic failure is injected with probability `STOCHASTIC_FAIL_PROB`.  
- Retries allowed: **N attempts**, base agent only.  
- If retries are disabled → run terminates early.

### Logical Failure
Evaluator detects incorrect or malformed step output.  
- Base runs: **no retries** (fail immediately).  
- R&R runs: **one retry** only using:
  - agent replacement  
  - prompt change (deep reasoning)  
  - parameter change  

If the retry fails → run stops.

---

## Datasets

The benchmark uses multi-step math datasets:

- GSM8K  
- GSM8K-Hard  
- ASDiv-A  
- AQuA-RAT  

Dataset loading is implemented in `datasets.py`.

---

## Output Metrics

For each benchmark condition, the system reports:

- Accuracy (%)  
- Average solve time (seconds)  
- Number of retries  
- Early termination counts  

All results are written to:

```text
logs/runs.jsonl
```

and visualized in the dashboard.

---

## Running the Benchmark

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure models
Local transformers:
```bash
export BASE_MODEL="your-small-model"
export LARGE_MODEL="your-large-model"
```

OpenAI:
```bash
export OPENAI_API_KEY="sk-..."
```

You can adjust settings in `config.py`.

### 3. Run benchmark
```bash
python -m project.benchmark --suite all --n 5
```

Results append automatically to `logs/runs.jsonl`.

---

## Dashboard

A Streamlit dashboard visualizes:

- Complete traces  
- Step attempts  
- Thinking (`<think>`) tokens  
- Answers (`<answer>`)  
- Evaluation decisions  
- Condition-level metrics  

Launch with:

```bash
streamlit run project/dashboard.py
```

---

## Trace Format

Each problem is logged as one JSON entry containing:

- Problem statement  
- Coordinator step plan  
- Step attempts  
- Evaluator outputs  
- Strategy used  
- Timing  
- Success/failure  

This enables detailed debugging and analysis.

---

## Project Structure

```text
project/
    agents.py
    coordinator.py
    evaluator.py
    orchestrator.py
    benchmark.py
    datasets.py
    dashboard.py
    config.py
logs/
    runs.jsonl
```

---

## Key Features

- Multi-agent collaborative reasoning  
- Controlled failure injection  
- Single-step logical recovery  
- Support for transformers and OpenAI  
- Deterministic trace logging  
- Visual analytics dashboard  
- Step-level timing and accuracy reporting  
