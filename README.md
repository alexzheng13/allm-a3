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

The original training and baseline scripts are unchanged.

## Run the Gemini planner

Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your shell. Do not store credentials in
source code or submit them. Then run:

~~~bash
python 2_llm_mcp.py
~~~

The script follows the original five STEP sections and executes directly at
module level. There is no main(), command-line argument parser or load_inputs().
Do not import it for inspection: importing runs the simulation.

It samples 100 combined cases using .sample(1) and relative data/model paths.
CASE_SEED defaults to 42 to make source-row sampling reproducible; it does not
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

VerifierAgent always runs; the confidence threshold defaults to 0.6:

1. VerifierAgent: the planner writes a review task into shared context. A second
   agent makes a separate Gemini call, seeing case features and the proposed
   decision but never true labels. It can retain or revise routes, change the
   primary pathway or abstain. Only reviewed decisions are sent to executors.
   It uses the same model with a different review prompt, not an independently
   trained medical expert; shared errors remain possible. An invalid or failed
   review blocks execution for that case and is recorded as failed.
2. Confidence gate: each selected pathway must have confidence >= 0.6. Low-score
   routes are removed. If the primary was removed, the remaining route with the
   highest confidence becomes primary. If no routes remain, status is abstained
   with abstention_reason=low_confidence. A model's own empty route list uses
   abstention_reason=model. These scores are not calibrated probabilities; 0.6 is
   an experimental policy setting, not a clinically established threshold.
3. Human-review queue: if every selected route is removed for low confidence,
   the case is marked pending_human_review and queued without waiting for input.
   These cases are written to human_review.csv at the end of the simulation.
   They are not sent to executors; other cases continue normally. Partial filtering
   and a model's own empty route list do not enter this queue. This implements a
   handoff for later review, not a completed human decision/execution loop.

make_case_text() is unchanged. Each of the three diseases must have a confidence score. The
same scheduler runs planner -> verifier -> confidence gate -> review queue or executors; the
agents exchange review and execution tasks through shared context.

No interactive input is requested. The review CSV includes case text, the proposed
routes and primary, all three confidence scores, threshold and decision notes, but
not true labels. human_routes, human_primary and human_notes are blank for later
manual review. Editing this CSV does not automatically execute classifiers; there
is no import/resume command. If no cases need review, the CSV has only its header.

Normal extended runs use about 200 Gemini calls for 100 cases (one planner and
one verifier call per case), before correction calls or API retries. Old results
without the extensions are not evidence of their performance. To compare confidence
filtering, keep CASE_SEED and the three disease pathways fixed:

~~~bash
# Full extended run (these are the defaults).
TRIAGE_CONFIDENCE_THRESHOLD=0.6 python 2_llm_mcp.py

# Same three disease pathways and verifier, without confidence filtering.
TRIAGE_CONFIDENCE_THRESHOLD=0 python 2_llm_mcp.py
~~~

All commands above make real API calls. CASE_SEED can be changed through the
environment for a new case sample. To run on a headless host, prefix a command
with MPLBACKEND=Agg; no backend is forced in the source code.

## Planner behavior

- make_case_text() is unchanged. Dataset labels are never sent to Gemini.
- JSON schema and Python validation restrict routes, primary, confidence and notes.
- Multiple routes and explicit abstention are allowed.
- Malformed output gets one correction call; a second failure is logged as
  parse_error with no routes and remains in the routing evaluation denominator.
- There is no model.api_error flag. Cases whose API requests exhaust retries have
  status failed in both logs, are skipped by executors, and are excluded from
  routing scores. Failure counts are printed separately. They are not retried in
  the next scheduler round. Ctrl+C interrupts normally and is not retried.
- Confidence values are uncalibrated model output, not diagnostic probabilities.

## Outputs

Each run creates a fresh results/gemini_.../ directory, preserving baseline files:

- executor_results.csv: routed classification results and true labels.
- llm_calls.jsonl: prompts, responses, parsing errors, timing and the calling agent.
- triage_decisions.json: final routes and status, original planner decision,
  verification decision/verdict, pre-threshold decision, filtered routes, threshold,
  abstention reason and human-review queue status when applicable,
  offline labels and source-row indices for reconstructing cases.
- human_review.csv: low-confidence cases awaiting manual review, with blank human
  decision columns and no ground-truth labels.
- confusion_diabetes.png, confusion_breast_cancer.png, confusion_heart.png:
  executor confusion matrices for the cases routed in this LLM run. Rows are true
  labels and columns are predictions, in label order 0, 1. Breast cancer uses
  0=malignant and 1=benign; the other two use 0=negative and 1=positive. Titles show
  the routed sample size. With no routed cases, the matrix is zero and marked
  "No routed cases" rather than interpreted as perfect classification.

Each run saves four data/log files and three PNG images. Routing metrics, executor
confusion matrices, case counts, runtime and token usage are also printed in the
terminal. No additional JSON, CSV or prompt files are generated, and results from
previous runs are not overwritten. The images measure classifier performance on
routed cases, not the LLM planner's routing precision or recall.

Routing precision is TP/(TP+FP); recall is TP/(TP+FN). The disease-positive label
is 1 for diabetes and heart, but 0 (malignant) for breast cancer. Undefined ratios
are zero for a nonempty evaluation; an empty evaluation reports null scores.
API/verification-failed, pending-human-review and unprocessed cases are excluded
and counted separately. A run with pending reviews is marked incomplete.
Abstentions remain in the routing denominator with no selected routes. Coverage
is the fraction of evaluated cases sent to at least one executor; failure counts
must also be reported rather than treating lower coverage as an improvement.
Pending-review counts are printed separately. Do not treat excluded uncertain cases
as correct predictions or claim that exporting a queue demonstrates completed human
review or clinical safety. The batch finishes, but summary.completed stays false
while there are unresolved human reviews.

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
