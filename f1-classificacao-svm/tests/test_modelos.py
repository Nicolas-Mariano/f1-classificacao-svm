import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

from f1_svm.__main__ import main
from f1_svm.avaliacao import analisar_discordancias, resumir_modelo
from f1_svm.experimento import executar
from f1_svm.modelos import PARAM_GRID, prever_regra_simples

GRADE_REDUZIDA = [{"kernel": ["rbf"], "C": [1, 10], "gamma": [0.1]}]


def test_grade_tem_72_combinacoes():
    assert len(ParameterGrid(PARAM_GRID)) == 72


def test_regra_simples_termina_onde_largou():
    previsoes = prever_regra_simples(pd.Series([1, 3, 4, 10, 11, 30]))

    assert previsoes.tolist() == [
        "podio",
        "podio",
        "pontua",
        "pontua",
        "fora_top10",
        "fora_top10",
    ]


def test_resume_as_metricas_de_um_modelo():
    real = ["podio", "pontua", "fora_top10", "fora_top10"]
    previsto = ["podio", "pontua", "fora_top10", "pontua"]

    resumo = resumir_modelo(real, previsto)

    assert resumo["acuracia"] == pytest.approx(0.75)
    assert resumo["f1_por_classe"]["podio"] == pytest.approx(1.0)
    assert resumo["previsoes_por_classe"] == {
        "podio": 1,
        "pontua": 2,
        "fora_top10": 1,
    }


def test_analisa_apenas_os_casos_em_que_os_modelos_discordam():
    real = np.array(["podio", "pontua", "fora_top10", "podio"])
    svm = np.array(["podio", "pontua", "pontua", "pontua"])
    regra = np.array(["podio", "podio", "fora_top10", "fora_top10"])

    analise = analisar_discordancias(real, svm, regra)

    assert analise["concordancia"] == pytest.approx(0.25)
    assert analise["casos_discordantes"] == 3
    assert analise["acertos_svm"] == 1
    assert analise["acertos_regra"] == 1
    assert analise["valor_p_mcnemar"] == pytest.approx(1.0)


def test_experimento_completo(resultados):
    resultado = executar(resultados, param_grid=GRADE_REDUZIDA, n_jobs=1)
    metricas = resultado.metricas

    assert metricas["linhas"]["teste"] == len(resultado.y_teste)
    assert metricas["temporadas"] == {"primeira": 2010, "ultima": 2026}
    assert set(metricas["modelos"]) == {
        "regra_simples",
        "svm_baseline",
        "svm_otimizado",
    }
    json.dumps(metricas)


def test_linha_de_comando_avisa_quando_faltam_dados(tmp_path, capsys):
    codigo = main(["--dados", str(tmp_path / "nao_existe.csv")])

    assert codigo == 1
    assert "data/README.md" in capsys.readouterr().err
