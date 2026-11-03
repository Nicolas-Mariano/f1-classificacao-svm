"""Leitura e preparação dos resultados de corrida.

Cada regra aqui corresponde a uma etapa do notebook. A ordem das operações
importa: as médias da temporada anterior precisam de `position` e de 2009,
que só são descartados depois de usados.
"""

from pathlib import Path

import pandas as pd

GRID_BOXES = 30
INICIO_ERA_MODERNA = 2010
INICIO_TESTE = 2024

COLUNAS_ESPARSAS = ["time", "fastest_lap", "fastest_lap_rank"]
COLUNAS_VAZAMENTO = ["position", "points", "laps", "status"]
COLUNAS_HISTORICO = ["media_piloto_ano_anterior", "media_equipe_ano_anterior"]
FEATURES = ["grid", *COLUNAS_HISTORICO]

# 'nome no dataset': 'nome único usado para a equipe inteira'
LINHAGEM_EQUIPES = {
    "brawn": "mercedes",  # Brawn GP (2009) virou Mercedes (2010)
    "bmw_sauber": "sauber",  # BMW Sauber -> Sauber
    "alfa": "sauber",  # Alfa Romeo (2019-2023) era a Sauber
    "audi": "sauber",  # Audi (2026) assumiu a Sauber
    "toro_rosso": "rb",  # Toro Rosso -> AlphaTauri -> RB
    "alphatauri": "rb",
    "force_india": "aston_martin",  # Force India -> Racing Point -> Aston Martin
    "racing_point": "aston_martin",
    "renault": "alpine",  # Renault -> Lotus F1 -> Renault -> Alpine
    "lotus_f1": "alpine",
    "virgin": "manor",  # Virgin -> Marussia -> Manor
    "marussia": "manor",
    "lotus_racing": "caterham",  # Lotus Racing (2010-2011) -> Caterham
}


def carregar_resultados(caminho: str | Path) -> pd.DataFrame:
    """Lê o `f1_results.csv` do dataset."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}. "
            "Veja em data/README.md como baixar o dataset."
        )
    return pd.read_csv(caminho)


def classificar_resultado(posicao: int) -> str:
    """Agrupa uma posição em pódio, zona de pontos ou fora do top 10."""
    if posicao <= 3:
        return "podio"
    if posicao <= 10:
        return "pontua"
    return "fora_top10"


def _media_ano_anterior(df: pd.DataFrame, chave: str, coluna: str) -> pd.DataFrame:
    """Calcula a chegada média por temporada e a desloca para o ano seguinte.

    A média de 2022 vira a informação usada nas corridas de 2023, então a
    feature já existe antes de qualquer corrida da temporada acontecer.
    """
    media = df.groupby(["season", chave])["position"].mean().reset_index()
    media = media.rename(columns={"position": coluna})
    media["season"] = media["season"] + 1
    return media


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza e cria o alvo e o histórico, da Etapa 3 à Etapa 11."""
    df = df.drop(columns=COLUNAS_ESPARSAS)

    # Grid 0 é largada dos boxes. Como número, seria lido como melhor que a pole.
    df["grid"] = df["grid"].replace(0, GRID_BOXES)

    df["resultado"] = df["position"].apply(classificar_resultado)
    df["equipe"] = df["constructor_id"].replace(LINHAGEM_EQUIPES)

    media_piloto = _media_ano_anterior(df, "driver_id", "media_piloto_ano_anterior")
    media_equipe = _media_ano_anterior(df, "equipe", "media_equipe_ano_anterior")
    df = df.merge(media_piloto, on=["season", "driver_id"], how="left")
    df = df.merge(media_equipe, on=["season", "equipe"], how="left")

    df = df.drop(columns=COLUNAS_VAZAMENTO)
    df = df[df["season"] >= INICIO_ERA_MODERNA].copy()
    return df.sort_values(["season", "round"]).reset_index(drop=True)


def dividir_por_temporada(
    df: pd.DataFrame, inicio_teste: int = INICIO_TESTE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e teste no tempo, para nunca usar o futuro."""
    treino = df[df["season"] < inicio_teste].copy()
    teste = df[df["season"] >= inicio_teste].copy()
    return treino, teste


def preencher_historico(
    treino: pd.DataFrame, teste: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Preenche quem não tem temporada anterior com a mediana do treino."""
    medianas = {coluna: float(treino[coluna].median()) for coluna in COLUNAS_HISTORICO}
    treino = treino.fillna(value=medianas)
    teste = teste.fillna(value=medianas)
    return treino, teste, medianas
