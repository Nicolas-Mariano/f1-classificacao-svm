"""Métricas e comparações entre os modelos."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

CLASSES = ["podio", "pontua", "fora_top10"]
NOMES_CLASSES = ["Pódio", "Pontua", "Fora do top 10"]


def resumir_modelo(y_real, y_previsto) -> dict:
    """Calcula as métricas usadas na comparação entre os modelos."""
    y_previsto = np.asarray(y_previsto)
    f1_classes = f1_score(y_real, y_previsto, labels=CLASSES, average=None)
    return {
        "f1_macro": float(f1_score(y_real, y_previsto, average="macro")),
        "acuracia": float(accuracy_score(y_real, y_previsto)),
        "f1_por_classe": {
            classe: float(valor) for classe, valor in zip(CLASSES, f1_classes)
        },
        "previsoes_por_classe": {
            classe: int((y_previsto == classe).sum()) for classe in CLASSES
        },
    }


def analisar_discordancias(y_real, y_svm, y_regra) -> dict:
    """Compara SVM e regra simples nos casos em que as previsões diferem.

    Quando as previsões diferem, no máximo uma está certa. O valor-p vem do
    teste de McNemar exato sobre os casos em que exatamente uma delas acertou.
    """
    y_real = np.asarray(y_real)
    y_svm = np.asarray(y_svm)
    y_regra = np.asarray(y_regra)

    diferentes = y_svm != y_regra
    acertos_svm = int((y_svm[diferentes] == y_real[diferentes]).sum())
    acertos_regra = int((y_regra[diferentes] == y_real[diferentes]).sum())
    decisivos = acertos_svm + acertos_regra
    valor_p = binomtest(acertos_svm, decisivos, 0.5).pvalue if decisivos else 1.0

    return {
        "concordancia": float((y_svm == y_regra).mean()),
        "casos_discordantes": int(diferentes.sum()),
        "acertos_svm": acertos_svm,
        "acertos_regra": acertos_regra,
        "valor_p_mcnemar": float(valor_p),
    }


def salvar_matriz_confusao(y_real, y_previsto, caminho: str | Path) -> None:
    """Salva a matriz de confusão como imagem."""
    figura, eixo = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_real,
        y_previsto,
        labels=CLASSES,
        display_labels=NOMES_CLASSES,
        cmap="Blues",
        colorbar=False,
        ax=eixo,
    )
    eixo.set_xlabel("Previsto pelo SVM")
    eixo.set_ylabel("Resultado real")
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
