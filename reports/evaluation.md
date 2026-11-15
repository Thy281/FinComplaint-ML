# Avaliação temporal — snapshot de 15/11/2026

Este relatório registra **métricas já existentes** no experimento MLflow local `model-received`; não é uma nova execução de treinamento ou avaliação. Os dados vieram da ABT local de 555.957 linhas, obtida do CSV com SHA-256 `df219bbcf0e8233f33eb83454ef9b3451adf471db29618acfb36d2dacc40ed3f`.

## Protocolo

`late_response = 1` significa que a empresa não respondeu no prazo. A divisão por `date_received` reserva as datas mais recentes para OOT e não divide um mesmo dia entre conjuntos. O ajuste e a busca temporal de hiperparâmetros usam apenas Treino; o script de avaliação carrega os runs concluídos e registra métricas de Teste/OOT neles. A classificação usa limiar `0,5`, enquanto Average Precision (AP) usa as probabilidades.

| Conjunto | Datas inclusivas | N | Positivos | Taxa positiva |
| --- | --- | ---: | ---: | ---: |
| Treino | 01/12/2011–28/04/2015 | 388.848 | 9.331 | 2,40% |
| Teste | 29/04/2015–18/10/2015 | 83.518 | 2.336 | 2,80% |
| OOT | 19/10/2015–25/04/2016 | 83.591 | 2.381 | 2,85% |

## Métricas registradas

| Run | Conjunto | AP | Precision | Recall | F1 | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BernoulliNB_original | Teste | 0,3832 | 0,1719 | 0,7449 | 0,2793 | 72.800 | 8.382 | 596 | 1.740 |
| BernoulliNB_original | OOT | 0,2685 | 0,1641 | 0,6287 | 0,2602 | 73.583 | 7.627 | 884 | 1.497 |
| XGBoost_over | Teste | 0,2432 | 0,2463 | 0,2937 | 0,2679 | 79.083 | 2.099 | 1.650 | 686 |

Run IDs dos registros mais recentes consultados: `BernoulliNB_original` = `253bde4ddf134e2e820aa0d9127cb947`; `XGBoost_over` = `bc25bb01677e4577acaa937d6fe9431f`. A AP média da validação cruzada temporal desses runs foi, respectivamente, `0,1340` e `0,1148`; as janelas da validação diferem do Teste e do OOT, portanto esses valores não são uma comparação pareada de conjuntos.

## Leitura dos resultados

- Para BernoulliNB, a AP cai de `0,383218` no Teste para `0,268476` no OOT: diferença absoluta de **−0,114742** e relativa de **−29,9%**. Precision, recall e F1 também caem no OOT com limiar `0,5`.
- O OOT possui 2,85% de positivos, mas isso não elimina a queda de AP. A diferença indica menor capacidade de ordenar casos de atraso no período mais recente; não identifica, por si só, a causa da mudança.
- `XGBoost_over` só tem métricas de Teste no fluxo atual. Não há base para afirmar seu desempenho no OOT ou escolher entre os dois modelos por generalização temporal.

## Limites de interpretação

O CSV é um snapshot histórico da base pública. Segundo o [CFPB](https://www.consumerfinance.gov/data-research/consumer-complaints/), reclamações são publicadas depois da resposta da empresa ou de 15 dias. A disponibilidade real das features no instante de `date_received` ainda precisa ser verificada em uma fonte operacional *point-in-time*. Além disso, o snapshot e os artefatos locais do MLflow não são versionados; para repetir exatamente estes números é necessário recuperar o mesmo CSV e o ambiente documentado no [README](../README.md).
