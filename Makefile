PYTHON = uv run --active python3
MAIN = src
FLAKE = uv run --active flake8
MYPY = uv run --active mypy
FLAGS = --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
PDB = $(PYTHON) -m pdb
DATA_PATH = data
LLM_PATH = llm_sdk

all: run

run:
	@$(PYTHON) -m $(MAIN)


install:
	@uv sync

lint:
	@$(FLAKE) $(MAIN)
	@$(MYPY) $(MAIN) $(FLAGS)

lint-strict:
	@$(FLAKE) $(MAIN)
	@$(MYPY) $(MAIN) --strict

debug:
	@$(PDB) $(MAIN)

clean:
	@rm -rf .mypy_cache
	@rm -rf __pycache__ $(MAIN)/__pycache__ $(LLM_PATH)/llm_sdk/__pycache__
	@rm -rf $(DATA_PATH)/output

.PHONY: all run install lint lint-strict debug clean