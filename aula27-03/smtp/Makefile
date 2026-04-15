# ─────────────────────────────────────────────────────
# SMTP — Standardized ML Training Pipeline
# Makefile para WSL (rodar via: wsl make <target>)
# ─────────────────────────────────────────────────────

PYTHON    := python.exe
MLFLOW    := $(PYTHON) -m mlflow
ENV       := --env-manager local

# Config padrão (pode ser sobrescrito: make train CONFIG=...)
CONFIG    := experimentos/classificacao_default.yaml

# ─── Targets Principais ──────────────────────────────

.PHONY: help install train classify regress tune ui clean

help: ## Mostra este menu de ajuda
	@echo ""
	@echo "  SMTP — Comandos disponíveis:"
	@echo "  ─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-12s %s\n", $$1, $$2}'
	@echo ""

install: ## Instala todas as dependências
	$(PYTHON) -m pip install mlflow scikit-learn pandas matplotlib seaborn pyyaml

train: ## Roda um experimento com CONFIG (default: classificacao_default)
	$(MLFLOW) run . $(ENV) -P config=$(CONFIG)

classify: ## Roda classificação padrão (50 épocas)
	$(MLFLOW) run . $(ENV) -P config=experimentos/classificacao_default.yaml

regress: ## Roda regressão padrão (80 épocas)
	$(MLFLOW) run . $(ENV) -P config=experimentos/regressao_default.yaml

tune: ## Roda classificação tunada (100 épocas, rede profunda)
	$(MLFLOW) run . $(ENV) -P config=experimentos/classificacao_tunada.yaml

ui: ## Abre o dashboard do MLflow (http://localhost:5000)
	$(MLFLOW) ui

list_exp: ## Lista todos os experimentos criados no MLflow
	$(MLFLOW) experiments search

clean: ## Remove artefatos temporários
	rm -rf artifacts_tmp
	@echo "Artefatos temporários removidos."
