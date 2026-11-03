"""Linha de comando: `python -m f1_svm`."""

import argparse
import json
import sys
from pathlib import Path

from f1_svm.avaliacao import salvar_matriz_confusao
from f1_svm.dados import carregar_resultados
from f1_svm.experimento import executar

NOMES_MODELOS = {
    "regra_simples": "Regra simples",
    "svm_baseline": "SVM baseline",
    "svm_otimizado": "SVM otimizado",
}


def montar_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m f1_svm",
        description="Treina os modelos e compara os resultados no teste.",
    )
    parser.add_argument(
        "--dados",
        default="data/raw/f1_results.csv",
        help="caminho do f1_results.csv (padrão: %(default)s)",
    )
    parser.add_argument(
        "--saida",
        default="resultados",
        help="pasta para métricas e figura (padrão: %(default)s)",
    )
    return parser


def imprimir_resumo(metricas: dict) -> None:
    linhas = metricas["linhas"]
    print(f"\nTreino: {linhas['treino']} linhas | Teste: {linhas['teste']} linhas")
    print(f"Melhores parâmetros: {metricas['melhores_parametros']}")
    print(f"F1 macro no treino:    {metricas['f1_macro_treino']:.4f}")
    print(f"F1 macro na validação: {metricas['f1_macro_validacao']:.4f}\n")

    print(f"{'Modelo':<16}{'F1 macro':>10}{'Acurácia':>10}")
    for chave, nome in NOMES_MODELOS.items():
        modelo = metricas["modelos"][chave]
        print(f"{nome:<16}{modelo['f1_macro']:>10.4f}{modelo['acuracia']:>10.4f}")

    discordancias = metricas["discordancias"]
    print(
        f"\nSVM e regra discordam em {discordancias['casos_discordantes']} casos: "
        f"SVM acertou {discordancias['acertos_svm']}, "
        f"regra acertou {discordancias['acertos_regra']} "
        f"(McNemar, p = {discordancias['valor_p_mcnemar']:.3f})."
    )


def main(argumentos: list[str] | None = None) -> int:
    opcoes = montar_argumentos().parse_args(argumentos)
    try:
        df = carregar_resultados(opcoes.dados)
    except FileNotFoundError as erro:
        print(erro, file=sys.stderr)
        return 1

    print("Treinando os modelos. A busca faz 360 treinos e pode levar alguns minutos.")
    resultado = executar(df)

    saida = Path(opcoes.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "metricas.json").write_text(
        json.dumps(resultado.metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    salvar_matriz_confusao(
        resultado.y_teste,
        resultado.previsoes_otimizado,
        saida / "matriz_confusao.png",
    )

    imprimir_resumo(resultado.metricas)
    print(f"\nArquivos salvos em {saida}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
