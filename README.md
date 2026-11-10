# Model Finance

Projeto de análise de reclamações de consumidores sobre produtos e serviços financeiros.

## Estrutura

```text
data/                 Arquivos de dados locais, ignorados pelo Git
model/                Scripts de análise e treinamento
reports/figures/      Gráficos produzidos pelas análises
requirements.txt      Dependências do ambiente Python
```

## Preparar o ambiente

Use Python 3.11 ou superior:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dados

Salve a fonte escolhida como `data/consumer_complaints.csv`. Os CSVs locais são ignorados pelo Git para evitar versionar arquivos grandes ou dados brutos por acidente. Registre a origem, a data de obtenção e as condições de uso do dataset antes de compartilhar resultados.

O arquivo analisado anteriormente tinha 555.957 registros, 18 colunas e o campo `consumer_disputed?`; ele descreve reclamações financeiras, não preços, transações nem inadimplência. Esse CSV não está presente atualmente na pasta `data/`, então os números precisam ser confirmados novamente quando o arquivo for colocado de volta.

## Análise exploratória

```bash
python model/01_eda.py
```

O script calcula ausências e distribuição do campo de contestação, além de salvar gráficos em `reports/figures/complaints_overview.png`. A variável-alvo de um futuro modelo deve ser escolhida a partir de uma pergunta de negócio explícita. Datas e respostas da empresa podem representar informações que ainda não existiam no momento de uma previsão e precisam ser avaliadas para evitar vazamento.
