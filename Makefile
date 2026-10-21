.PHONY: synth up down llm llm14 pareto-finance-all judge-finance preflight eval-all clean-finance smoke-eval-finance-14b pareto-finance oracle-finance oracle-finance-14b ingest-finance smoke-eval-finance smoke smoke-rag smoke-agent smoke-eval smoke-eval-14b pareto engine ingest eval route-train deploy dash demo test
MODEL ?= mlx-community/Qwen2.5-7B-Instruct-4bit
MODEL14 ?= mlx-community/Qwen2.5-14B-Instruct-4bit
MODES ?= rag,agent   # arms to run; run one at a time overnight, e.g. MODES=agent
RESUME ?=            # RESUME=--resume continues an interrupted run (same PER/MODES/prompt only)
EXCLUDE ?=           # EXCLUDE=metrics-generated skips numeric finance questions (D39)
PER ?= 10   # questions per type for eval targets; PER=5 for a quick run

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
judge-finance: ; for f in eval/runs/smoke_eval_finance.json eval/runs/smoke_eval_finance_14b.json; do PYTHONPATH=. uv run python eval/judge.py $$f --model $(MODEL14) --base-url http://localhost:8081/v1; done
# After `make judge-finance`: headline finance table with judge scores (numeric questions excluded, D38).
pareto-finance-judged: ; PYTHONPATH=. uv run python eval/pareto.py eval/runs/smoke_eval_finance.judged.json eval/runs/smoke_eval_finance_14b.judged.json --exclude metrics-generated | tee eval/runs/pareto_finance_judged.txt
# Judge validation: export 30 stratified rows, hand-fill the human_label column, then score agreement/kappa.
judge-sample: ; PYTHONPATH=. uv run python eval/judge_validate.py export eval/runs/smoke_eval_finance.judged.json
judge-score: ; PYTHONPATH=. uv run python eval/judge_validate.py score eval/runs/judge_sample.csv
pareto:    ; PYTHONPATH=. uv run python eval/pareto.py eval/runs/smoke_eval.json eval/runs/smoke_eval_14b.json
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
