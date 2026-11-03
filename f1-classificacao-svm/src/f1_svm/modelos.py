"""Modelos comparados: regra simples, SVM baseline e SVM otimizado."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from f1_svm.dados import FEATURES, classificar_resultado

# Duas grades porque o kernel linear não usa gamma: 8 + 64 = 72 combinações.
PARAM_GRID = [
    {
        "kernel": ["linear"],
        "C": [0.1, 1, 10, 100],
        "class_weight": ["balanced", None],
    },
    {
        "kernel": ["rbf", "sigmoid"],
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
        "class_weight": ["balanced", None],
    },
]


def padronizar(treino: pd.DataFrame, teste: pd.DataFrame):
    """Padroniza as features com um escalador ajustado só no treino."""
    escalador = StandardScaler()
    X_treino = escalador.fit_transform(treino[FEATURES])
    X_teste = escalador.transform(teste[FEATURES])
    return X_treino, X_teste, escalador


def treinar_baseline(X_treino, y_treino) -> SVC:
    """Treina o SVM de referência, com kernel RBF e classes balanceadas."""
    modelo = SVC(kernel="rbf", class_weight="balanced")
    return modelo.fit(X_treino, y_treino)


def buscar_melhor_svm(
    X_treino, y_treino, param_grid=None, n_jobs: int = -1
) -> GridSearchCV:
    """Busca hiperparâmetros validando sempre em dados mais recentes."""
    busca = GridSearchCV(
        SVC(),
        PARAM_GRID if param_grid is None else param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="f1_macro",
        n_jobs=n_jobs,
    )
    return busca.fit(X_treino, y_treino)


def prever_regra_simples(grid: pd.Series) -> pd.Series:
    """Prevê que cada piloto termina na mesma faixa em que largou."""
    return grid.apply(classificar_resultado)
