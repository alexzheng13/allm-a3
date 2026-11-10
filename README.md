# Assignment 3: hospital routing simulation

An educational simulation for diabetes, breast cancer and heart disease.
The LLM planner selects routes, the verifier reviews them, and random-forest
executors classify the routed cases.

## 1. Install dependencies

Run all commands from the project directory.

```bash
conda create -n allm-a3 python=3.10 pip -y
conda activate allm-a3
python -m pip install -r requirements.txt
```

## 2. Configure Gemini

```bash
export GEMINI_API_KEY="your-api-key"
```

The default model is `gemini-3.1-flash-lite`. To use another model:

```bash
export GEMINI_MODEL="your-model-id"
```

Do not submit your API key. A normal run uses about 200 API calls for
100 cases, excluding corrections and retries, and may incur charges.

## 3. Run the scripts in order

```bash
(
  set -e
  export MPLBACKEND=Agg
  mkdir -p model results
  python -u 0_train_models.py
  python -u 1_basic_mcp.py
  python -u 2_llm_mcp.py
)
```

- `0_train_models.py`: trains and saves the three classifiers in `model/`.
- `1_basic_mcp.py`: runs the rule-based baseline.
- `2_llm_mcp.py`: runs the Gemini planner, verifier and executors.

Both simulations sample 100 cases with seed 42. Gemini uses temperature 0.2;
its responses are not guaranteed to repeat across runs.

## 4. Results

The baseline saves `baseline_predictions.csv`, `baseline_metrics.csv`,
`output.png` and three `confusion_*.png` images under `results/`.

Each LLM run creates a new `results/gemini_<timestamp>/` containing:

- `executor_results.csv`: executor predictions and true labels.
- `llm_calls.jsonl`: planner/verifier prompts, responses and call logs.
- `triage_decisions.json`: original and reviewed routing decisions.
- Three `confusion_*.png` images: executor confusion matrices.

Metrics are also printed in the terminal. Rerunning training and the baseline
overwrites their existing outputs; LLM runs use separate timestamped folders.

The cases combine records from different people, and classifiers are evaluated
on their training datasets. These results are not evidence of clinical validity.
