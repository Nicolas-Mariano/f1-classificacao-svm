"""Executa o experimento completo, do CSV à comparação dos modelos."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from f1_svm.avaliacao import analisar_discordancias, resumir_modelo
from f1_svm.dados import dividir_por_temporada, preencher_historico, preparar_dados
from f1_svm.modelos import (
    buscar_melhor_svm,
    padronizar,
    prever_regra_simples,
    treinar_baseline,
)


@dataclass
class Resultado:
    """Métricas do experimento e previsões usadas para gerá-las."""

    metricas: dict
    y_teste: pd.Series
    previsoes_regra: np.ndarray
    previsoes_baseline: np.ndarray
    previsoes_otimizado: np.ndarray


def executar(df: pd.DataFrame, param_grid=None, n_jobs: int = -1) -> Resultado:
    """Treina os modelos e os avalia nas temporadas de teste."""
    dados = preparar_dados(df)
    treino, teste = dividir_por_temporada(dados)
    treino, teste, medianas = preencher_historico(treino, teste)

    X_treino, X_teste, _ = padronizar(treino, teste)
    y_treino, y_teste = treino["resultado"], teste["resultado"]

    baseline = treinar_baseline(X_treino, y_treino)
    busca = buscar_melhor_svm(X_treino, y_treino, param_grid, n_jobs)
    melhor = busca.best_estimator_

    previsoes_regra = prever_regra_simples(teste["grid"]).to_numpy()
    previsoes_baseline = baseline.predict(X_teste)
    previsoes_otimizado = melhor.predict(X_teste)

    metricas = {
        "linhas": {"treino": len(treino), "teste": len(teste)},
        "temporadas": {
            "primeira": int(dados["season"].min()),
            "ultima": int(dados["season"].max()),
        },
        "medianas": medianas,
        "melhores_parametros": busca.best_params_,
        "f1_macro_validacao": float(busca.best_score_),
        "f1_macro_treino": float(
            f1_score(y_treino, melhor.predict(X_treino), average="macro")
        ),
        "modelos": {
            "regra_simples": resumir_modelo(y_teste, previsoes_regra),
            "svm_baseline": resumir_modelo(y_teste, previsoes_baseline),
            "svm_otimizado": resumir_modelo(y_teste, previsoes_otimizado),
        },
        "discordancias": analisar_discordancias(
            y_teste, previsoes_otimizado, previsoes_regra
        ),
    }
    return Resultado(
        metricas, y_teste, previsoes_regra, previsoes_baseline, previsoes_otimizado
    )
