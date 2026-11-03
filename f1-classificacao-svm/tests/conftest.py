"""Dados sintéticos com o mesmo formato do `f1_results.csv`.

Os testes não dependem do dataset real, que não é versionado.
"""

import numpy as np
import pandas as pd
import pytest


def equipe_na_temporada(linhagem: list[tuple[int, str]], temporada: int) -> str:
    """Devolve o nome que uma equipe usava naquela temporada."""
    nome = linhagem[0][1]
    for inicio, candidato in linhagem:
        if temporada >= inicio:
            nome = candidato
    return nome


LINHAGENS = [
    [(0, "force_india"), (2019, "racing_point"), (2021, "aston_martin")],
    [(0, "toro_rosso"), (2020, "alphatauri"), (2024, "rb")],
    [(0, "renault"), (2012, "lotus_f1"), (2016, "renault"), (2021, "alpine")],
    [(0, "bmw_sauber"), (2010, "sauber"), (2019, "alfa"), (2024, "sauber")],
    [(0, "brawn"), (2010, "mercedes")],
    [(0, "ferrari")],
    [(0, "red_bull")],
    [(0, "mclaren")],
    [(0, "williams")],
    [(2016, "haas")],
]


def gerar_resultados(
    temporadas=range(2008, 2027), corridas: int = 5, semente: int = 7
) -> pd.DataFrame:
    """Gera resultados em que o grid e a força do piloto explicam a chegada."""
    rng = np.random.default_rng(semente)
    forca_equipe = {i: rng.normal(0, 1.2) for i in range(len(LINHAGENS))}
    proximo_piloto = 0
    pilotos: dict[tuple[int, int], tuple[str, float]] = {}
    linhas = []

    for temporada in temporadas:
        ativos = [
            i for i, linhagem in enumerate(LINHAGENS) if temporada >= linhagem[0][0]
        ]
        for equipe in ativos:
            for carro in (0, 1):
                if (equipe, carro) not in pilotos or rng.random() < 0.15:
                    pilotos[(equipe, carro)] = (
                        f"piloto_{proximo_piloto}",
                        rng.normal(0, 1),
                    )
                    proximo_piloto += 1

        for rodada in range(1, corridas + 1):
            grade = [(e, c) for e in ativos for c in (0, 1)]
            ritmo = np.array(
                [
                    forca_equipe[e] + pilotos[(e, c)][1] + rng.normal(0, 0.8)
                    for e, c in grade
                ]
            )
            ordem_grid = (-ritmo).argsort().argsort() + 1
            corrida = ritmo + rng.normal(0, 1.1, len(grade))
            ordem_chegada = (-corrida).argsort().argsort() + 1

            for indice, (equipe, carro) in enumerate(grade):
                piloto = pilotos[(equipe, carro)][0]
                construtor = equipe_na_temporada(LINHAGENS[equipe], temporada)
                grid = int(ordem_grid[indice])
                if rng.random() < 0.02:
                    grid = 0
                posicao = int(ordem_chegada[indice])
                linhas.append(
                    {
                        "season": temporada,
                        "round": rodada,
                        "race_name": f"GP {rodada}",
                        "driver_id": piloto,
                        "driver_name": piloto.replace("_", " ").title(),
                        "constructor_id": construtor,
                        "constructor": construtor.title(),
                        "grid": grid,
                        "position": posicao,
                        "points": max(0.0, 11.0 - posicao),
                        "laps": 58,
                        "status": "Finished",
                        "time": None,
                        "fastest_lap": None,
                        "fastest_lap_rank": None,
                    }
                )

    resultados = pd.DataFrame(linhas)
    return resultados.sort_values(["season", "round", "position"]).reset_index(
        drop=True
    )


@pytest.fixture
def resultados() -> pd.DataFrame:
    return gerar_resultados()
