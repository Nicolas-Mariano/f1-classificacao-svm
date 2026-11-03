# Dados

O projeto usa o arquivo `f1_results.csv` do dataset
[Formula 1 Complete Dataset 1950-2026](https://www.kaggle.com/datasets/patelris/formula-1-complete-dataset-1950-2026),
de `patelris`, disponível no Kaggle. Ele não é versionado aqui.

1. Baixe o dataset pela página do Kaggle.
2. Salve o arquivo como `data/raw/f1_results.csv`.

Os números do notebook e do pôster usam a versão com 25.939 linhas. Versões mais
novas do dataset podem trazer mais corridas e mudar os resultados.

## Colunas usadas

| Coluna | Uso |
| --- | --- |
| `season`, `round` | Ordem cronológica e divisão entre treino e teste |
| `driver_id` | Cálculo do histórico do piloto |
| `constructor_id` | Cálculo do histórico da equipe, com os nomes unificados |
| `grid` | Posição de largada, usada como feature |
| `position` | Criação do alvo e do histórico; removida antes do treino |
