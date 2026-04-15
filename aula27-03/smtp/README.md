# SMTP

Pipeline padronizado para treinamento supervisionado com rastreio via MLflow.

## Ambiente recomendado

- Python 3.12.
- No Windows, prefira criar uma venv nova com Python 3.12 antes de executar o projeto.

## O que esta atividade contempla

- Portabilidade do notebook original para a estrutura modular em `smtp`.
- Uso de dataset no processo de treinamento com `scikit-learn`.
- Execução de pelo menos dois experimentos.
- Registro de parâmetros, métricas e artefatos no MLflow.
- Acompanhamento dos treinamentos pelo dashboard do MLflow.
- Preparação do projeto para publicação em um repositório GitHub.

## Experimentos e perguntas objetivas

- `experimentos/classificacao_default.yaml`: este experimento responde se uma MLP padrao ja separa bem os casos de classificacao usando uma configuracao base.
- `experimentos/classificacao_tunada.yaml`: este experimento responde se ajustar profundidade, dropout e early stopping melhora a classificacao em relacao ao baseline.
- `experimentos/regressao_default.yaml`: este experimento responde como a mesma pipeline se comporta em regressao no dataset de diabetes com configuracao padrao.

A Run 1 apresentou Accuracy 0.9211, F1 0.9209 e ROC-AUC 0.9728 por utilizar uma configuracao que se ajustou melhor ao dataset, com arquitetura [100, 50], ativacao relu e solver adam, resultando em melhor generalizacao.

Ja a Run 2, no mesmo dataset, teve desempenho inferior, com Accuracy 0.6228, F1 0.4780 e ROC-AUC 0.3402.
Uma explicacao tecnica plausivel e que a combinacao de rede mais profunda [128, 64, 32], ativacao tanh, solver sgd e early stopping nao generalizou tao bem neste cenario.
Em resumo: mais tuning nao garantiu melhor resultado; neste caso, a configuracao baseline foi superior.
## Estrutura

- `src/preprocess.py`: ingestão e split treino/validação/teste.
- `src/train.py`: treino, avaliação e logging no MLflow.
- `src/utils.py`: métricas, gráficos e artefatos.
- `experimentos/`: configs YAML dos experimentos.
- `MLproject`: entrada para execução com `mlflow run`.

## Como executar

Antes de rodar os experimentos, crie e ative uma venv com Python 3.12 e instale as dependências:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install mlflow
```

Depois, dentro da pasta `smtp`:

```powershell
python -m mlflow run . --env-manager local -P config=experimentos/classificacao_default.yaml
python -m mlflow run . --env-manager local -P config=experimentos/classificacao_tunada.yaml
python -m mlflow run . --env-manager local -P config=experimentos/regressao_default.yaml
```

Se o comando `python` do ambiente apontar para a venv ativada, essa forma evita o erro de `mlflow` nao reconhecido no PATH.

Se quiser comparar apenas a classificacao, rode os dois primeiros comandos. Isso atende ao requisito de pelo menos dois experimentos distintos, com configuracoes separadas.

## Acompanhar no MLflow

```powershell
python -m mlflow ui --backend-store-uri sqlite:///C:/Users/Usuario/Documents/S.i/6/IA/ia/aula27-03/smtp/mlflow.db
```

Depois, abrir o endereço exibido pelo comando e comparar parâmetros, métricas e artefatos das execuções. Se a interface anterior estiver aberta, feche e abra de novo com esse backend-store-uri para carregar os runs de 14/04.

## Publicação no GitHub

Se o repositório ainda não tiver remoto configurado:

```powershell
git init
git add .
git commit -m "Porta notebook para pipeline com MLflow"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

## Saídas esperadas

- Artefatos de matriz de confusão, ROC, PR, curva de treino e importância de variáveis.
- Métricas finais no MLflow para comparação entre experimentos.
- Modelo registrado no MLflow Model Registry.
