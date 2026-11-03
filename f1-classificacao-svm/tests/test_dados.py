import pandas as pd
import pytest

from f1_svm.dados import (
    COLUNAS_VAZAMENTO,
    FEATURES,
    GRID_BOXES,
    carregar_resultados,
    classificar_resultado,
    dividir_por_temporada,
    preencher_historico,
    preparar_dados,
)


def linha(season, driver_id, constructor_id, grid, position):
    return {
        "season": season,
        "round": 1,
        "race_name": "GP",
        "driver_id": driver_id,
        "driver_name": driver_id,
        "constructor_id": constructor_id,
        "constructor": constructor_id,
        "grid": grid,
        "position": position,
        "points": 0.0,
        "laps": 50,
        "status": "Finished",
        "time": None,
        "fastest_lap": None,
        "fastest_lap_rank": None,
    }


@pytest.mark.parametrize(
    ("posicao", "esperado"),
    [
        (1, "podio"),
        (3, "podio"),
        (4, "pontua"),
        (10, "pontua"),
        (11, "fora_top10"),
        (39, "fora_top10"),
    ],
)
def test_classifica_os_limites_de_cada_faixa(posicao, esperado):
    assert classificar_resultado(posicao) == esperado


def test_largada_dos_boxes_vai_para_o_fim_do_grid():
    df = pd.DataFrame([linha(2010, "ana", "ferrari", 0, 5)])

    assert preparar_dados(df)["grid"].tolist() == [GRID_BOXES]


def test_equipe_renomeada_mantem_o_historico():
    df = pd.DataFrame(
        [
            linha(2020, "ana", "racing_point", 1, 4),
            linha(2021, "bia", "aston_martin", 1, 8),
        ]
    )

    dados = preparar_dados(df)
    aston_2021 = dados[dados["season"] == 2021].iloc[0]

    assert aston_2021["media_equipe_ano_anterior"] == 4


def test_historico_usa_apenas_a_temporada_anterior():
    df = pd.DataFrame(
        [
            linha(2021, "ana", "ferrari", 1, 2),
            linha(2021, "ana", "ferrari", 1, 4),
            linha(2022, "ana", "ferrari", 1, 20),
        ]
    )

    dados = preparar_dados(df)
    ana_2022 = dados[dados["season"] == 2022].iloc[0]

    assert ana_2022["media_piloto_ano_anterior"] == 3


def test_resultado_da_propria_temporada_nao_altera_as_features(resultados):
    alterados = resultados.copy()
    temporada_final = alterados["season"] == 2026
    alterados.loc[temporada_final, "position"] = 1

    originais = preparar_dados(resultados)
    modificados = preparar_dados(alterados)

    pd.testing.assert_frame_equal(originais[FEATURES], modificados[FEATURES])


def test_estreante_fica_sem_historico_ate_o_preenchimento():
    df = pd.DataFrame([linha(2015, "novata", "ferrari", 5, 6)])

    dados = preparar_dados(df)

    assert dados["media_piloto_ano_anterior"].isna().all()


def test_colunas_com_vazamento_sao_removidas(resultados):
    dados = preparar_dados(resultados)

    assert not set(COLUNAS_VAZAMENTO) & set(dados.columns)
    assert not set(COLUNAS_VAZAMENTO) & set(FEATURES)


def test_mantem_apenas_a_era_moderna(resultados):
    dados = preparar_dados(resultados)

    assert dados["season"].min() == 2010


def test_treino_vem_sempre_antes_do_teste(resultados):
    treino, teste = dividir_por_temporada(preparar_dados(resultados))

    assert treino["season"].max() < teste["season"].min()
    assert teste["season"].min() == 2024


def test_medianas_nao_usam_o_teste():
    treino = pd.DataFrame(
        {
            "media_piloto_ano_anterior": [2.0, 4.0, None],
            "media_equipe_ano_anterior": [6.0, 8.0, 10.0],
        }
    )
    teste = pd.DataFrame(
        {
            "media_piloto_ano_anterior": [None, 100.0],
            "media_equipe_ano_anterior": [None, 100.0],
        }
    )

    treino, teste, medianas = preencher_historico(treino, teste)

    assert medianas == {
        "media_piloto_ano_anterior": 3.0,
        "media_equipe_ano_anterior": 8.0,
    }
    assert teste["media_piloto_ano_anterior"].tolist() == [3.0, 100.0]
    assert not treino.isna().any().any()


def test_arquivo_ausente_explica_como_obter_os_dados(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        carregar_resultados(tmp_path / "nao_existe.csv")
