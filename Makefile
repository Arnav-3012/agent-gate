.PHONY: up down llm smoke smoke-rag engine ingest eval route-train deploy dash demo test
MODEL ?= mlx-community/Qwen2.5-7B-Instruct-4bit

up:        ; docker compose up -d qdrant
down:      ; docker compose down
llm:       ; uv run mlx_lm.server --model $(MODEL) --port 8080
smoke:     ; PYTHONPATH=. uv run python scripts/smoke_llm.py
engine:    ; uv run uvicorn engine.api.main:app --port 8000 --reload
ingest:    ; PYTHONPATH=. uv run python scripts/ingest.py
smoke-rag: ; PYTHONPATH=. uv run python scripts/smoke_rag.py
eval:      ; uv run python eval/run_baselines.py && uv run python eval/benchmark.py
route-train: ; uv run python eval/build_router_labels.py && uv run python router_model/train.py
deploy:    ; cd infra && uv run cdk deploy --all --require-approval never
dash:      ; cd dashboard && npm run dev
demo:      ; @echo "TODO Phase 7: warm-up + up + llm + engine + dash"
test:      ; uv run pytest -q
