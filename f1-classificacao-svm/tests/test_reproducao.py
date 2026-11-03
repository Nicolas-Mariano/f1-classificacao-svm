"""Confere se o código reproduz os números do notebook e do pôster.

Precisa do dataset real em `data/raw/f1_results.csv`, então é ignorado na
integração contínua. Se o dataset ganhar novas corridas, os valores mudam.
"""

from pathlib import Path

import pytest

from f1_svm.dados import carregar_resultados
from f1_svm.experimento import executar

CAMINHO_DADOS = Path(__file__).parents[1] / "data" / "raw" / "f1_results.csv"


@pytest.mark.skipif(not CAMINHO_DADOS.exists(), reason="dataset real não encontrado")
def test_reproduz_os_numeros_do_notebook():
    df = carregar_resultados(CAMINHO_DADOS)
    assert len(df) == 25939, "versão do dataset diferente da usada no notebook"

    metricas = executar(df).metricas
    modelos = metricas["modelos"]

    assert metricas["linhas"] == {"treino": 5953, "teste": 1024}
    assert metricas["melhores_parametros"] == {
        "C": 10,
        "class_weight": None,
        "gamma": 0.1,
        "kernel": "rbf",
    }
    assert metricas["f1_macro_validacao"] == pytest.approx(0.664433, abs=1e-5)
    assert modelos["regra_simples"]["f1_macro"] == pytest.approx(0.717258, abs=1e-5)
    assert modelos["svm_baseline"]["f1_macro"] == pytest.approx(0.669444, abs=1e-5)
    assert modelos["svm_otimizado"]["f1_macro"] == pytest.approx(0.704052, abs=1e-5)
    assert metricas["discordancias"]["casos_discordantes"] == 144
    assert metricas["discordancias"]["acertos_svm"] == 65
    assert metricas["discordancias"]["acertos_regra"] == 75
