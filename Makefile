# One-command reproducibility.
#   make test            unit tests (no model needed)
#   make run K=5         run every task K times -> runs/main_k$(K).jsonl
#   make eval            metrics + failure taxonomy for that run
#   make judge           LLM-judge audit (full and reply views)
#   make probes          control + rename + reformat + misleading runs (~90 min)
#   make probe-report    compare the probes against the main run
#   make all             everything above, in order

PY      ?= python
MODEL   ?= qwen3.5:4b-mlx
JUDGE   ?= gemma4:e4b
K       ?= 5
RUN     := runs/main_k$(K).jsonl
BENCH   := $(PY) -m agent.run_bench --model $(MODEL) --k $(K)

.PHONY: help install test reset run eval judge probes probe-report all

help:
	@sed -n '2,8p' Makefile

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

reset:
	$(PY) -m env.db

run: reset
	$(BENCH) --out $(RUN) --resume

eval:
	$(PY) -m eval.metrics $(RUN)
	$(PY) -m eval.failures $(RUN)

judge:
	$(PY) -m eval.judge $(RUN) --view full --model $(JUDGE) --resume
	$(PY) -m eval.judge $(RUN) --view reply --model $(JUDGE) --resume

probes: reset
	$(BENCH) --seed-base 100 --out runs/control_k$(K).jsonl --resume
	$(BENCH) --variant rename --out runs/probe_rename_k$(K).jsonl --resume
	$(BENCH) --variant reformat --out runs/probe_reformat_k$(K).jsonl --resume
	$(BENCH) --variant misleading --out runs/probe_misleading_k$(K).jsonl --resume

probe-report:
	$(PY) -m eval.probes $(RUN) \
		control=runs/control_k$(K).jsonl \
		rename=runs/probe_rename_k$(K).jsonl \
		reformat=runs/probe_reformat_k$(K).jsonl \
		misleading=runs/probe_misleading_k$(K).jsonl

all: test run eval judge probes probe-report
