import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_diabetes

def load_data(problem_type="classification"):
    """Simula a ingestão de dados carregando um dataset padrão"""
    if problem_type == "classification":
        data = load_breast_cancer()
    else:
        data = load_diabetes()
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data.feature_names

def preprocess_and_split(X, y, test_size=0.2, val_size=0.2, random_state=42, problem_type="classification"):
    """
    Realiza o split dos dados.
    Divide inicialmente em treino/teste com val_size somado.
    Em seguida separa a validação mantendo a proporção correta, mandatório pela RF01 do PRD.
    """
    val_ratio = val_size / (test_size + val_size)
    stratify_target = y if problem_type == "classification" else None
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state
        , stratify=stratify_target
    )
    
    stratify_temp = y_temp if problem_type == "classification" else None
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=stratify_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
