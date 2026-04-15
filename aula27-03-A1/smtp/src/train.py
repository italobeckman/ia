import argparse
import os
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import log_loss, mean_squared_error
import numpy as np
from preprocess import load_data, preprocess_and_split
import utils
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Carrega a configuração de um experimento a partir de um arquivo YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuração carregada de: {config_path}")
    return config


def build_model(problem_type, model_cfg, optim_cfg, train_cfg):
    """
    Constrói o modelo MLP a partir das seções 'model', 'optimizer' e 'training' do YAML.
    Mapeia todos os hiperparâmetros suportados pelo sklearn MLP.
    """
    hidden_layers = tuple(model_cfg.get("hidden_layers", [100, 50]))
    activation    = model_cfg.get("activation", "relu")
    solver        = model_cfg.get("solver", "adam")
    alpha         = model_cfg.get("alpha", 0.0001)
    batch_size    = model_cfg.get("batch_size", "auto")
    seed          = train_cfg.get("seed", 42)
    shuffle       = train_cfg.get("shuffle", True)

    lr_init       = optim_cfg.get("learning_rate", 0.001)
    lr_schedule   = optim_cfg.get("learning_rate_schedule", "constant")
    momentum      = optim_cfg.get("momentum", 0.9)
    nesterovs     = optim_cfg.get("nesterovs_momentum", True)
    beta_1        = optim_cfg.get("beta_1", 0.9)
    beta_2        = optim_cfg.get("beta_2", 0.999)
    epsilon       = optim_cfg.get("epsilon", 1e-8)

    common_params = dict(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=lr_schedule,
        learning_rate_init=lr_init,
        momentum=momentum,
        nesterovs_momentum=nesterovs,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        shuffle=shuffle,
        random_state=seed,
        max_iter=1,
        warm_start=True,
    )

    if problem_type == "classification":
        model = MLPClassifier(**common_params)
    else:
        model = MLPRegressor(**common_params)

    return model


def train_with_epoch_logging(model, X_train, y_train, X_val, y_val,
                             n_epochs, problem_type, early_stopping, patience, tolerance):
    """
    Treina o MLP época por época via partial_fit e loga métricas no MLflow.
    Suporta early stopping com patience e tolerance configuráveis.
    """
    train_losses, val_losses = [], []
    train_scores, val_scores = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.partial_fit(X_train, y_train)

        if problem_type == "classification":
            train_loss = log_loss(y_train, model.predict_proba(X_train))
            val_loss   = log_loss(y_val,   model.predict_proba(X_val))
            train_acc  = model.score(X_train, y_train)
            val_acc    = model.score(X_val,   y_val)
            train_score = train_acc
            val_score = val_acc
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "train_accuracy": train_acc,
                "val_accuracy":   val_acc,
            }, step=epoch)
        else:
            train_loss = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
            val_loss   = np.sqrt(mean_squared_error(y_val,   model.predict(X_val)))
            train_r2   = model.score(X_train, y_train)
            val_r2     = model.score(X_val,   y_val)
            train_score = train_r2
            val_score = val_r2
            mlflow.log_metrics({
                "train_rmse": train_loss,
                "val_rmse":   val_loss,
                "train_r2":   train_r2,
                "val_r2":     val_r2,
            }, step=epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_scores.append(train_score)
        val_scores.append(val_score)

        if epoch % 10 == 0:
            logger.info(f"Época {epoch}/{n_epochs} — train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Early stopping
        if early_stopping:
            if val_loss < best_val_loss - tolerance:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping na época {epoch} (patience={patience})")
                mlflow.log_metric("stopped_epoch", epoch)
                break

    mlflow.log_metric("total_epochs_trained", epoch)
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_score": train_scores,
        "val_score": val_scores,
    }


def main(config_path):
    config = load_config(config_path)

    # ── Extrair seções do YAML ──
    problem_type    = config["problem_type"]
    experiment_name = config.get("experiment_name", f"SMTP_{problem_type.capitalize()}")
    experiment_question = config.get("experiment_question")

    data_cfg  = config.get("data", {})
    model_cfg = config.get("model", {})
    optim_cfg = config.get("optimizer", {})
    train_cfg = config.get("training", {})

    test_size      = data_cfg.get("test_size", 0.2)
    val_size       = data_cfg.get("val_size", 0.2)
    n_epochs       = train_cfg.get("n_epochs", 50)
    seed           = train_cfg.get("seed", 42)
    early_stopping = train_cfg.get("early_stopping", False)
    patience       = train_cfg.get("patience", 10)
    tolerance      = train_cfg.get("tolerance", 1e-4)

    utils.set_seed(seed)

    logger.info(f"Experimento: {experiment_name} | Tipo: {problem_type}")

    # ── Pipeline: ingestão → pré-processamento → split ──
    X, y, feature_names = load_data(problem_type)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_and_split(
        X, y, test_size=test_size, val_size=val_size, random_state=seed, problem_type=problem_type
    )
    logger.info(f"Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]} | Teste: {X_test.shape[0]}")

    # ── Construir modelo a partir do YAML ──
    model = build_model(problem_type, model_cfg, optim_cfg, train_cfg)

    if problem_type == "classification":
        classes = np.unique(y_train)
        model.partial_fit(X_train, y_train, classes=classes)
    else:
        model.partial_fit(X_train, y_train)

    # ── MLflow run context ──
    env_run_id = os.environ.get("MLFLOW_RUN_ID")
    if env_run_id:
        run_context = mlflow.start_run(run_id=env_run_id)
    else:
        mlflow.set_experiment(experiment_name)
        run_context = mlflow.start_run()

    with run_context:
        # Log de TODOS os hiperparâmetros (flat) para comparação fácil no dashboard
        mlflow.log_params({
            "config_file":            os.path.basename(config_path),
            "problem_type":           problem_type,
            "experiment_question":    experiment_question or "",
            # Data
            "data.test_size":         test_size,
            "data.val_size":          val_size,
            # Model
            "model.hidden_layers":    str(model_cfg.get("hidden_layers", [100, 50])),
            "model.activation":       model_cfg.get("activation", "relu"),
            "model.solver":           model_cfg.get("solver", "adam"),
            "model.alpha":            model_cfg.get("alpha", 0.0001),
            "model.batch_size":       str(model_cfg.get("batch_size", "auto")),
            # Optimizer
            "optim.learning_rate":    optim_cfg.get("learning_rate", 0.001),
            "optim.lr_schedule":      optim_cfg.get("learning_rate_schedule", "constant"),
            "optim.momentum":         optim_cfg.get("momentum", 0.9),
            "optim.nesterovs":        optim_cfg.get("nesterovs_momentum", True),
            "optim.beta_1":           optim_cfg.get("beta_1", 0.9),
            "optim.beta_2":           optim_cfg.get("beta_2", 0.999),
            "optim.epsilon":          optim_cfg.get("epsilon", 1e-8),
            # Training
            "train.n_epochs":         n_epochs,
            "train.seed":             seed,
            "train.early_stopping":   early_stopping,
            "train.patience":         patience,
            "train.tolerance":        tolerance,
            "train.shuffle":          train_cfg.get("shuffle", True),
        })
        # Salvar o YAML como artefato
        mlflow.log_artifact(config_path, artifact_path="config")

        logger.info(f"Treinando por até {n_epochs} épocas (early_stopping={early_stopping})...")
        history = train_with_epoch_logging(
            model, X_train, y_train, X_val, y_val,
            n_epochs, problem_type, early_stopping, patience, tolerance
        )

        os.makedirs("artifacts_tmp", exist_ok=True)
        history_path = os.path.join("artifacts_tmp", "training_history.png")
        utils.plot_training_history_and_save(
            history["train_loss"],
            history["val_loss"],
            history["train_score"],
            history["val_score"],
            history_path,
            score_label="Accuracy" if problem_type == "classification" else "R2 Score",
        )
        mlflow.log_artifact(history_path)

        logger.info("Avaliando no conjunto de teste...")
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            metrics = utils.calculate_classification_metrics(y_test, y_pred, y_prob)

            cm_path = os.path.join("artifacts_tmp", "confusion_matrix.png")
            utils.plot_confusion_matrix_and_save(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path)

            if y_prob is not None:
                roc_path = os.path.join("artifacts_tmp", "roc_curve.png")
                pr_path  = os.path.join("artifacts_tmp", "pr_curve.png")
                r, p = utils.plot_roc_and_pr_curves_and_save(y_test, y_prob, roc_path, pr_path)
                if r and p:
                    mlflow.log_artifact(r)
                    mlflow.log_artifact(p)
        else:
            metrics = utils.calculate_regression_metrics(y_test, y_pred)
            res_path = os.path.join("artifacts_tmp", "residuals.png")
            utils.plot_regression_residuals_and_save(y_test, y_pred, res_path)
            mlflow.log_artifact(res_path)

        mlflow.log_metrics(metrics)
        logger.info(f"Métricas finais: {metrics}")

        # Feature importance via importâncias nativas ou permutation importance
        fi_path = os.path.join("artifacts_tmp", "feature_importance.png")
        if utils.plot_feature_importance_and_save(model, feature_names, fi_path, X_reference=X_val, y_reference=y_val):
            mlflow.log_artifact(fi_path)

        # Model Registry
        signature     = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=f"SMTP_{problem_type.capitalize()}_Model"
        )

        logger.info("Modelo logado com sucesso no MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="experimentos/classificacao_default.yaml",
        help="Caminho do arquivo YAML de configuração do experimento"
    )
    args = parser.parse_args()
    main(args.config)
