# Assignment 3: hospital routing simulation

The Gemini planner chooses diagnostic pathways. The supplied random forests
classify routed cases. This is an educational simulation, not a medical decision
system or a deployed MCP protocol server.

## Setup

Use Python 3.10 or newer. Run every command from the assignment directory:

~~~bash
conda create -n allm-a3 python=3.10 pip -y
conda activate allm-a3
python -m pip install -r requirements.txt
python 0_train_models.py
mkdir -p results
python 1_basic_mcp.py
~~~

The training script uses RandomForestClassifier(random_state=42, n_jobs=-1) for
all three diseases. Training overwrites the three model/*.pkl files; reproducible
predictions also depend on using the same data and software versions. It still
trains on the full datasets, without a held-out test split.

The baseline keeps the original rule-based planners and executors, but fixes
sampling with np.random.seed(42), uses unique IDs patient_0000 through patient_0099,
creates results/ automatically and saves prediction/metric CSV files. Each of the
100 IDs has three independently sampled domain records, giving 300 diagnostic
tasks rather than 300 distinct patients.

## Run the Gemini planner

Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your shell. Do not store credentials in
source code or submit them. Then run:

~~~bash
python 2_llm_mcp.py
~~~

To rerun the complete pipeline after configuring your key:

~~~bash
(
  set -e
  export MPLBACKEND=Agg
  python -u 0_train_models.py
  python -u 1_basic_mcp.py
  python -u 2_llm_mcp.py
)
~~~

Training overwrites the model files. The baseline overwrites its same-named files
under results/; the Gemini simulation creates a new timestamped directory.

The script follows the original five STEP sections and executes directly at
module level. There is no main(), command-line argument parser or load_inputs().
Do not import it for inspection: importing runs the simulation.

It samples 100 combined cases using .sample(1) and relative data/model paths.
random_seed is fixed at 42 to make source-row sampling reproducible; it does not
make Gemini responses deterministic. The constructor retains the six original
arguments and the three original disease pathways. The original
executor is unchanged. Short UUID-based IDs are checked for duplicates so that
cases cannot overwrite another case's review or executor task.

The default model is gemini-3.1-flash-lite, overridable with GEMINI_MODEL.
call_llm_fn() creates and closes its own client each time. Generation uses
temperature 0.2, at most 2048 output tokens and a one-second pause per request.
SDK retries are disabled. Actual API calls consume project quota and may incur
charges depending on the project's billing plan.

If a generation request fails, call_llm_fn() waits five seconds and retries up to
three times (four attempts including the initial request). A request that still
fails is recorded as failed for that case; the remaining cases continue.

## Optional extensions

VerifierAgent is the only retained bonus extension and always runs. The planner
writes a review task into shared context. The verifier makes a separate Gemini
call, seeing case features and the proposed decision but never true labels. It
can retain or revise the selected routes and primary. Only valid reviewed
decisions are sent to executors. An invalid or failed review blocks execution
for that case and is recorded as failed. The verifier uses the same model with
a different review prompt, not an independently trained expert; shared errors
remain possible.

The scheduler runs planner -> verifier -> executors, exchanging tasks through
shared context. make_case_text() is unchanged. Each response must select one
or more routes, with primary among them. Confidence values remain in the original
JSON format for logging only: there is no confidence threshold, explicit
abstention option or human-review queue. Low confidence does not prevent a valid
route from being executed, so this educational policy is not a clinical safeguard.

Normal runs use about 200 Gemini calls for 100 cases (one planner and one
verifier call per case), before correction calls or API retries. Any older results
from the threshold/human-review version describe that version, not this one.

Running the script makes real API calls. The sampling seed is set in the source,
not through an environment variable. To run on a headless host, prefix a command
with MPLBACKEND=Agg; no backend is forced in the source code.

## Planner behavior

- make_case_text() is unchanged. Dataset labels are never sent to Gemini.
- JSON schema and Python validation restrict routes, primary, confidence and notes.
- One or more routes are required. Empty routes or a null primary are invalid
  responses and get the same correction handling as malformed JSON.
- Malformed output gets one correction call; a second failure is logged as
  parse_error with no routes and remains in the routing evaluation denominator.
- There is no model.api_error flag. Cases whose API requests exhaust retries have
  status failed in both logs, are skipped by executors, and are excluded from
  routing scores. Failure counts are printed separately. They are not retried in
  the next scheduler round. Ctrl+C interrupts normally and is not retried.
- Confidence values are uncalibrated model output, not diagnostic probabilities.
  They are recorded but do not filter routes.

## Outputs

The baseline writes these files directly under results/:

- baseline_predictions.csv: individual predictions, labels and domain IDs.
- baseline_metrics.csv: per-disease accuracy and disease-positive precision,
  recall and F1; breast cancer uses positive label 0 (malignant).
- output.png: prediction counts by disease.
- confusion_diabetes.png, confusion_breast_cancer.png, confusion_heart.png:
  baseline executor confusion matrices. These are not the LLM-routed matrices.

Each Gemini run creates a fresh results/gemini_.../ directory, preserving baseline files:

- executor_results.csv: routed classification results and true labels.
- llm_calls.jsonl: prompts, responses, parsing errors, timing and the calling agent.
- triage_decisions.json: final routes and status, original planner decision,
  verification decision/verdict,
  offline labels and source-row indices for reconstructing cases.
- confusion_diabetes.png, confusion_breast_cancer.png, confusion_heart.png:
  executor confusion matrices for the cases routed in this LLM run. Rows are true
  labels and columns are predictions, in label order 0, 1. Breast cancer uses
  0=malignant and 1=benign; the other two use 0=negative and 1=positive. Titles show
  the routed sample size. With no routed cases, the matrix is zero and marked
  "No routed cases" rather than interpreted as perfect classification.

Each run saves three data/log files and three PNG images. Routing metrics, executor
confusion matrices, case counts, runtime and token usage are also printed in the
terminal. No additional JSON, CSV or prompt files are generated, and results from
previous runs are not overwritten. The images measure classifier performance on
routed cases, not the LLM planner's routing precision or recall.

The verifier-only run in [results/gemini_20261109_223524_589924/](results/gemini_20261109_223524_589924/)
contains 100 cases, all with status ok, and 232 executed diagnostic tasks. Its
llm_calls.jsonl contains 100 planner calls and 100 verifier calls, with no logged
API failures or parsing corrections. The verifier kept routes/primary unchanged
for 50 cases and changed them for 50; this count alone does not show that the
changes improved routing. It is a single run, not a statistical comparison.

Routing precision is TP/(TP+FP); recall is TP/(TP+FN). The disease-positive label
is 1 for diabetes and heart, but 0 (malignant) for breast cancer. Undefined ratios
are zero for a nonempty evaluation; an empty evaluation reports null scores.
API/verification-failed and unprocessed cases are excluded and counted separately.
Planner parse failures remain in the routing denominator with no selected routes;
these are technical failures, not intentional abstentions. Coverage is the fraction
of evaluated cases sent to at least one executor. Failure counts must also be
reported rather than treating exclusions as an improvement.

Executor scores are conditional on routing. The three original classifiers are
still evaluated in-sample because their training data are also used for sampling.
Combined features come from
different people, and disease labels are not a gold standard for ordering tests.
These are software-routing metrics, not evidence of clinically valid triage.

## Syntax check

~~~bash
python -m py_compile 0_train_models.py 1_basic_mcp.py 2_llm_mcp.py
~~~

This checks Python syntax without running simulations or sending API requests.
The current project directory does not include the earlier offline test suite.
Run 0_train_models.py first if the three model files need to be regenerated.

## References

- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini 3.1 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
