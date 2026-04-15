Este documento descreve os requisitos para a criação de um framework padronizado de **Engenharia de Machine Learning (MLE)** focado em aprendizado supervisionado. O objetivo é garantir que todo o ciclo de vida do modelo seja rastreável, escalável e rigorosamente avaliado.

---

## 1. Visão Geral e Propósito
**Nome do Projeto:** Standardized ML Training Pipeline (SMTP)
**Propósito:** Estabelecer uma infraestrutura robusta e modular para o desenvolvimento, treinamento e avaliação de modelos de IA supervisionados, utilizando o **MLflow** como espinha dorsal para versionamento e governança, garantindo que experimentos sejam 100% reprodutíveis e comparáveis.

---

## 2. Requisitos Funcionais

### RF01: Desenvolvimento e Treinamento Supervisionado
* **Modularização:** O pipeline deve ser dividido em módulos claros: `data_ingestion`, `preprocessing`, `feature_engineering`, `train` e `evaluate`.
* **Flexibilidade de Algoritmos:** Suporte para modelos de Classificação e Regressão (ex: PyTorch, Scikit-learn, XGBoost).
* **Split de Dados:** Implementação mandatória de divisão entre Treino, Validação e Teste (ou Cross-Validation).

### RF02: Avaliação e Métricas Profissionais
O sistema deve calcular automaticamente as métricas baseadas no tipo de problema:

| Tipo de Problema | Métricas Obrigatórias |
| :--- | :--- |
| **Classificação** | Accuracy, Precision, Recall, F1-Score, ROC-AUC. |
| **Regressão** | MAE, MSE, $R^2$ Score, RMSE. |

### RF03: Visualização e Gráficos de Performance
O pipeline deve gerar e salvar como artefatos os seguintes gráficos:
* **Para Classificação:** Matriz de Confusão, Curva ROC e Precision-Recall Curve.
* **Para Regressão:** Gráfico de Resíduos (Errors vs. Predicted) e Distribuição de Erros.
* **Para a rodada de treinamento:** Gráfico de evolução das métricas ao longo das épocas ou iterações contendo a métrica de treino e validação.
* **Geral:** Gráfico de Importância de Variáveis (Feature Importance).

---

## 3. Gestão de Ciclo de Vida com MLflow

Para atender à estrutura de Engenharia de ML, o uso do MLflow será dividido em três pilares:

1.  **MLflow Tracking:**
    * **Parâmetros:** Logar automaticamente hiperparâmetros (ex: `learning_rate`, `depth`).
    * **Métricas:** Logar a evolução das métricas a cada época ou iteração.
    * **Artefatos:** Salvar o arquivo `.pkl` ou `.onnx` do modelo, além dos gráficos gerados em PNG/HTML.
2.  **MLflow Projects:** Utilizar arquivos `MLproject` para definir o ambiente (Conda ou Docker) e garantir a execução consistente em qualquer máquina.
3.  **Model Registry:** Interface para transição de estágios do modelo (*Staging*, *Production*, *Archived*).


---

## 4. Estrutura de Projeto (MLE Best Practices)

A estrutura de diretórios deve seguir o padrão de engenharia de software para ML:

```text
├── data/               # Dados brutos e processados (não versionados no Git)
├── models/             # Local de salvamento temporário de modelos
├── notebooks/          # Exploração inicial (EDAs)
├── src/                # Código fonte modular
│   ├── train.py        # Script principal de treino
│   ├── preprocess.py   # Limpeza e transformação
│   └── utils.py        # Funções de ajuda e gráficos
├── tests/              # Testes unitários para o pipeline
├── MLproject           # Configuração de execução do MLflow
└── pyproject.toml      # Gestão de dependências (Poetry/Pip)
```

---

## 5. Requisitos Não Funcionais
* **Reprodutibilidade:** O uso de *seeds* fixas em todas as bibliotecas (NumPy, Pandas, modelos) é obrigatório.
* **Escalabilidade:** O código de pré-processamento deve ser eficiente para lidar com volumes crescentes de dados (vetorização em vez de loops).
* **Observabilidade:** Logs detalhados em tempo de execução para facilitar o debug de falhas no treinamento.

---

## 6. Critérios de Aceite
* O experimento é iniciado via comando `mlflow run`.
* As métricas e gráficos aparecem corretamente no Dashboard do MLflow.
* O modelo final é registrado no *Model Registry* com sua respectiva assinatura de entrada/saída.
* O cálculo de erro quadrático médio (RMSE) deve seguir a fórmula:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
