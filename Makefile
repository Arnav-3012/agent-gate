.PHONY: armset-b armset-b-judge phase3 phase3-report phase3-7b phase3-14b paired fixtest-recall fixtest-t2 fixtest-t3 fixtest-report synth up down llm llm14 pareto-finance-all judge-finance preflight eval-all clean-finance smoke-eval-finance-14b pareto-finance oracle-finance oracle-finance-14b ingest-finance smoke-eval-finance smoke smoke-rag smoke-agent smoke-eval smoke-eval-14b pareto engine ingest eval route-train deploy dash demo test
MODEL ?= mlx-community/Qwen2.5-7B-Instruct-4bit
MODEL14 ?= mlx-community/Qwen2.5-14B-Instruct-4bit
MODES ?= rag,agent   # arms to run; run one at a time overnight, e.g. MODES=agent
RESUME ?=            # RESUME=--resume continues an interrupted run (same PER/MODES/prompt only)
EXCLUDE ?=           # EXCLUDE=metrics-generated skips numeric finance questions (D39)
JUDGE ?= local   # judge-finance: local (Qwen-14B, free, self-preference) | hosted (Haiku via ANTHROPIC_API_KEY, D42)
PER ?= 10   # questions per type for eval targets; PER=5 for a quick run

# Fix test (D43-D45): T1 needs only Qdrant; T2/T3 need the 7B server (:8080). Outputs eval/runs/fixtest_*.json.
fixtest-recall: ; PYTHONPATH=. uv run python -m eval.recall_at_k && PYTHONPATH=. uv run python -m eval.recall_at_k --domain general --per-type 10
fixtest-t2: ; PYTHONPATH=. uv run python -m eval.run_baselines --domain finance --per-type 10 --modes rag --out eval/runs/fixtest_t2_rag.json --resume
fixtest-t3: ; PYTHONPATH=. uv run python -m eval.run_baselines --domain finance --per-type 5 --modes agent --max-tokens 6000 --out eval/runs/fixtest_t3_agent6000.json --resume && PYTHONPATH=. uv run python -m eval.run_baselines --domain finance --per-type 5 --modes agent --max-tokens 9000 --out eval/runs/fixtest_t3_agent9000.json --resume
paired: ; PYTHONPATH=. uv run python -m eval.paired eval/runs/baseline_*.json
fixtest-report: ; PYTHONPATH=. uv run python -m eval.fixtest_report eval/runs/fixtest_t2_rag.json eval/runs/fixtest_t3_agent6000.json eval/runs/fixtest_t3_agent9000.json

# Phase 3 main eval (D43-D45 config frozen in config/engine.yaml). One arm per process, all resumable:
# `make llm` (7B :8080) then `make phase3-7b`; later stop it, `make llm14` (:8081) then `make phase3-14b`.
# general = 300 Qs (100 per type); finance = 100 text Qs (50 per type, numeric skipped, D39).
P3 = PYTHONPATH=. uv run python -m eval.run_baselines --resume
phase3-7b:
	$(P3) --domain general --per-type 100 --modes rag
	$(P3) --domain finance --per-type 50 --modes rag
	$(P3) --domain general --per-type 100 --modes agent
	$(P3) --domain finance --per-type 50 --modes agent
# Arm set B (D52): cross-tier hosted arms + closed-book 7B. Caps sum to $2.80 (expected ~$1.9); key: set -a; source .env; set +a
HOSTED = $(P3) --provider anthropic --modes rag
armset-b:
	$(P3) --domain general --per-type 100 --modes closed --out eval/runs/baseline_general_closed.json
	$(HOSTED) --model claude-haiku-5-5 --tag haiku --domain general --per-type 100 --max-usd 0.15
	$(HOSTED) --model claude-haiku-5-5 --tag haiku --domain finance --per-type 50 --max-usd 0.15
	$(HOSTED) --model claude-sonnet-5-5 --tag sonnet --domain general --per-type 100 --max-usd 1.10
	$(HOSTED) --model claude-sonnet-5-5 --tag sonnet --domain finance --per-type 50 --max-usd 0.90
armset-b-judge:
	for f in general_closed general_haiku general_sonnet; do PYTHONPATH=. uv run python eval/judge.py eval/runs/baseline_$$f.json --domain general --judge hosted --max-usd 0.10; done
	for f in finance_haiku finance_sonnet; do PYTHONPATH=. uv run python eval/judge.py eval/runs/baseline_$$f.json --judge hosted --max-usd 0.10; done
phase3-14b:
	$(P3) --model $(MODEL14) --base-url http://localhost:8081/v1 --tag 14b --domain general --per-type 100 --modes rag
	$(P3) --model $(MODEL14) --base-url http://localhost:8081/v1 --tag 14b --domain finance --per-type 50 --modes rag
	$(P3) --model $(MODEL14) --base-url http://localhost:8081/v1 --tag 14b --domain general --per-type 100 --modes agent
	$(P3) --model $(MODEL14) --base-url http://localhost:8081/v1 --tag 14b --domain finance --per-type 50 --modes agent

# One command for the whole Phase 3 main eval, in order: needs Qdrant + BOTH servers up (`make up`, `make llm`,
# `make llm14` in two other terminals; same as eval-all). Re-run `make phase3` after any interruption: --resume skips
# finished (question, arm) pairs. Analysis output lands in eval/runs/phase3_*.txt. The hosted judge is NOT in here (needs a key + go).
S7 = default
S14 = Qwen2.5-14B-Instruct-4bit
phase3: preflight phase3-7b phase3-14b phase3-report
phase3-report:
	PYTHONPATH=. uv run python -m eval.paired eval/runs/baseline_*.json | tee eval/runs/phase3_paired.txt
	PYTHONPATH=. uv run python -m eval.benchmark eval/runs/baseline_*.json | tee eval/runs/phase3_benchmark.txt
	PYTHONPATH=. uv run python -m eval.cascade_cv eval/runs/baseline_*.json --domain general --cheap $(S7)/rag --strong $(S14)/rag | tee eval/runs/phase3_cascade_general.txt
	PYTHONPATH=. uv run python -m eval.cascade_cv eval/runs/baseline_*.json --domain finance --cheap $(S7)/rag --strong $(S14)/rag | tee eval/runs/phase3_cascade_finance.txt

up:        ; docker compose up -d qdrant
down:      ; docker compose down
llm:       ; uv run mlx_lm.server --model $(MODEL) --port 8080
llm14:     ; uv run mlx_lm.server --model $(MODEL14) --port 8081
smoke:     ; PYTHONPATH=. uv run python scripts/smoke_llm.py
smoke-eval: ; PYTHONPATH=. uv run python scripts/smoke_eval.py --per-type $(PER)
smoke-eval-14b: ; PYTHONPATH=. uv run python scripts/smoke_eval.py --per-type $(PER) --model $(MODEL14) --base-url http://localhost:8081/v1 --tag 14b
ingest-finance: ; PYTHONPATH=. uv run python scripts/ingest_finance.py
smoke-eval-finance: ; PYTHONPATH=. uv run python scripts/smoke_eval.py --domain finance --per-type $(PER) --modes $(MODES) $(RESUME) $(if $(EXCLUDE),--exclude-types $(EXCLUDE))
smoke-eval-finance-14b: ; PYTHONPATH=. uv run python scripts/smoke_eval.py --domain finance --per-type $(PER) --modes $(MODES) $(RESUME) $(if $(EXCLUDE),--exclude-types $(EXCLUDE)) --model $(MODEL14) --base-url http://localhost:8081/v1 --tag 14b
# Headline finance table excludes numeric questions (D38); the appendix table keeps everything.
pareto-finance: ; PYTHONPATH=. uv run python eval/pareto.py eval/runs/smoke_eval_finance.json eval/runs/smoke_eval_finance_14b.json --exclude metrics-generated
pareto-finance-all: ; PYTHONPATH=. uv run python eval/pareto.py eval/runs/smoke_eval_finance.json eval/runs/smoke_eval_finance_14b.json
oracle-finance: ; PYTHONPATH=. uv run python scripts/smoke_eval.py --domain finance --per-type $(PER) --modes oracle,rag --tag oracle
oracle-finance-14b: ; PYTHONPATH=. uv run python scripts/smoke_eval.py --domain finance --per-type $(PER) --modes oracle,rag --model $(MODEL14) --base-url http://localhost:8081/v1 --tag oracle14b
preflight:
	@curl -sf localhost:8080/v1/models >/dev/null || { echo "7B server down: make llm"; exit 1; }
	@curl -sf localhost:8081/v1/models >/dev/null || { echo "14B server down: make llm14"; exit 1; }
	@curl -sf localhost:6333/collections >/dev/null || { echo "Qdrant down: make up"; exit 1; }
clean-finance: ; rm -f eval/runs/smoke_eval_finance*.json
# One command for the whole Phase 2 closing run, in order (est. 1.5-2 h). Re-run with
# `make eval-all RESUME=--resume` after an interruption: finished (question, arm) pairs are skipped.
eval-all: preflight
	$(MAKE) ingest-finance
	$(MAKE) oracle-finance PER=5
	$(MAKE) smoke-eval-finance PER=20
	$(MAKE) smoke-eval-finance-14b PER=20
	$(MAKE) smoke-eval-14b PER=10
	$(MAKE) pareto | tee eval/runs/pareto_general.txt
	$(MAKE) pareto-finance | tee eval/runs/pareto_finance.txt
	$(MAKE) pareto-finance-all | tee eval/runs/pareto_finance_all.txt
judge-finance: ; for f in eval/runs/baseline_finance.json eval/runs/baseline_finance_14b.json; do PYTHONPATH=. uv run python eval/judge.py $$f --judge $(JUDGE) $(if $(filter local,$(JUDGE)),--model $(MODEL14) --base-url http://localhost:8081/v1); done
# After `make judge-finance`: headline finance table with judge scores (numeric questions excluded, D38).
pareto-finance-judged: ; PYTHONPATH=. uv run python eval/pareto.py eval/runs/baseline_finance.judged.json eval/runs/baseline_finance_14b.judged.json --exclude metrics-generated | tee eval/runs/pareto_finance_judged.txt
# Judge validation: export 30 stratified rows, hand-fill the human_label column, then score agreement/kappa.
judge-sample: ; PYTHONPATH=. uv run python eval/judge_validate.py export eval/runs/baseline_finance.judged.json
judge-score: ; PYTHONPATH=. uv run python eval/judge_validate.py score eval/runs/judge_sample.csv
pareto:    ; PYTHONPATH=. uv run python eval/pareto.py eval/runs/smoke_eval.json eval/runs/smoke_eval_14b.json
export BUDGET_EMAIL := $(shell sed -n 's/^BUDGET_EMAIL=\([^ ]*\).*/\1/p' .env 2>/dev/null)
export ENGINE_KEY := $(shell sed -n 's/^ENGINE_KEY=\([^ ]*\).*/\1/p' .env 2>/dev/null)
engine:    ; uv run uvicorn engine.api.main:app --port 8000 --reload
ingest:    ; PYTHONPATH=. uv run python scripts/ingest.py
smoke-rag: ; PYTHONPATH=. uv run python scripts/smoke_rag.py
smoke-agent: ; PYTHONPATH=. uv run python scripts/smoke_agent.py
eval:      ; uv run python eval/run_baselines.py && uv run python eval/benchmark.py
route-train: ; uv run python eval/build_router_labels.py && uv run python router_model/train.py
umap:      ; PYTHONPATH=. uv run python scripts/umap_export.py   # Qdrant up; writes dashboard/public/data/galaxy.json
synth:     ; uv run --group infra python infra/app.py   # template only, no AWS calls
deploy:    ; cd infra && npx aws-cdk deploy --all --require-approval never   # needs Arnav's go + spend check
dash:      ; cd dashboard && npm run dev
demo:      ; @echo "TODO Phase 7: warm-up + up + llm + engine + dash"
test:      ; uv run pytest -q
