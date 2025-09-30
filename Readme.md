# Análise Comparativa de Técnicas de Machine Learning para Predição de Indicadores da COVID-19

Este projeto implementa uma análise comparativa de diferentes técnicas de Machine Learning para predição de indicadores da COVID-19, considerando múltiplos modelos e análise de importância de variáveis.

## Configuração do Ambiente

### Requisitos do Sistema
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Configuração do Ambiente Virtual

1. Clone o repositório:
```bash
git clone git@github.com:BetoFigu66/Prj_TCC_JRSF_MBA_USP_ESALQ.git
cd Prj_TCC_JRSF_MBA_USP_ESALQ
```

2. Crie um ambiente virtual:

No Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

No Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Verifique a instalação:
```bash
python -c "import numpy; import pandas; import sklearn; import matplotlib; import seaborn; print('Todas as dependências foram instaladas com sucesso')"
```

## Download e Organização dos Dados (obrigatório)

Como este repositório não inclui dados públicos, você precisará baixá-los e organizá-los localmente na pasta `DataSources/` antes de executar a carga e o treinamento.

1) Dados obrigatórios para executar `Scripts/carga_dados.py`

- COVID-19 (caso_full)
  - Fonte: Brasil.io (Séries históricas COVID-19 por município)
  - Link: https://brasil.io/dataset/covid19/caso_full/
    Clique em "BAIXAR DADOS COMPLETOS EM CSV" e depois em caso_full.csv.gz	
  - Arquivo esperado: 'caso_full.csv.gz' `caso_full.csv`
  - Descompacte em: `DataSources/caso_full_csv/caso_full.csv`

- Eleições municipais 2016 (prefeitos eleitos)
  - Fonte: TSE (Portal do TSE)
  - Link: https://sig.tse.jus.br/ords/dwapr/r/seai/sig-eleicao
    Escolher Conjunto de dados / Candidaturas / Candidatos
      Filtros:
          Cargo: Prefeito
          Tipo de eleição: Ordinária
        Dimensões: Todas
        Métricas: Quantidade de candidatos eleitos
  - Após o download: extraia o `.zip` e você obterá `candidatos.csv` (delimitador `;`, codificação latin1)
  - Renomeie para: `votacao_candidato_2016.csv` (para compatibilizar com `Scripts/carga_dados.py`)
  - Coloque em: `DataSources/votacao_candidato/votacao_candidato_2016.csv`

- Características Municipais (IBGE MUNIC)
  - Fonte: IBGE — Pesquisa de Informações Básicas Municipais (MUNIC)
  - Link: https://ftp.ibge.gov.br/Perfil_Municipios/2020/Base_de_Dados/
  - Exemplo usado: `Base_MUNIC_2020.xlsx`
  - Coloque em: `DataSources/IBGE/Base_MUNIC_2020.xlsx`

- PIB dos Municípios (2010–2021)
  - Fonte: IBGE — Contas Nacionais, PIB dos Municípios
  - Link: https://ftp.ibge.gov.br/Pib_Municipios/2020/base/base_de_dados_2010_2020_xls.zip
  - Exemplo usado: `PIB dos Municípios - base de dados 2010-2020.xls` (ou CSV equivalente)
  - Coloque em: `DataSources/IBGE_Economicos/PIB_municipios_2010_2021.xlsx`

3) Estrutura local esperada (resumo)

```
Projeto_TCC_Jun_25/
└── DataSources/
    ├── caso_full_csv/
    │   └── caso_full.csv
    ├── votacao_candidato/
    │   └── votacao_candidato_2016.csv
    ├── IBGE/
    │   └── Base_MUNIC_2020.xlsx            
    ├── IBGE_Economicos/
        └── PIB_municipios_2010_2021.xlsx   
```

Com isso, você poderá executar:

```bash
python Scripts/main.py
```

## Estrutura do Projeto

```
Prj_TCC_JRSF_MBA_USP_ESALQ/
├── Scripts/
│   ├── main.py                        # Interface principal do sistema
│   ├── carga_dados.py                 # Processamento inicial dos dados
│   ├── visualization.py               # Geração de gráficos e visualizações
│   ├── models_to_train.json           # Configuração dos modelos a serem treinados
│   ├── compare_random_forest_models.py # Comparação IBGE vs baseline
│   ├── compare_pib_models.py          # Comparação PIB vs baseline
│   ├── ibge_data_explorer.py          # Exploração dados IBGE
│   └── pib_data_explorer.py           # Exploração dados PIB
├── Models/
│   ├── models.py                      # Biblioteca principal de ML
│   └── trainedModels/                 # Modelos treinados e logs
├── Data/
│   └── Processed/                     # Dados processados
├── DataSources/                       # Dados originais
│   ├── caso_full_csv/                 # Dados COVID-19
│   ├── votacao_candidato/             # Dados eleitorais TSE
│   ├── IBGE/                          # Dados IBGE (MUNIC e PIB)
│   └── BR_UF_2024/                    # Shapefiles geográficos
├── Visualizacoes/                     # Gráficos gerados
├── Docs/                              # Documentação do projeto
├── requirements.txt                   # Dependências do projeto
└── README.md                         # Este arquivo
```

## Fontes de Dados

### Dados COVID-19
- Casos confirmados, óbitos e métricas relacionadas por município
- Fonte: Brasil.io — https://brasil.io/dataset/covid19/caso_full/

### Dados Municipais
- Características municipais, indicadores socioeconômicos e infraestrutura de saúde
- Fontes: IBGE (MUNIC) — https://www.ibge.gov.br/estatisticas/sociais/saude/10586-pesquisa-de-informacoes-basicas-municipais.html

### Dados Eleitorais
- Perfil dos prefeitos eleitos, partido político, características demográficas
- Fonte: TSE — Repositório de Dados Abertos (https://dadosabertos.tse.jus.br/) e CDN (https://cdn.tse.jus.br/)

## Processamento de Dados

O processamento dos dados é realizado através dos scripts em `Scripts/`, que incluem:
- Limpeza e normalização
- Junção das bases
- Cálculo de indicadores
- Preparação para modelagem

## Modelos de Machine Learning

O projeto utiliza quatro tipos principais de modelos:

1. DecisionTreeRegressor (Árvore de Decisão)
   - Configurável via parâmetros max_depth, min_samples_split, min_samples_leaf
   
2. Random Forest
   - Suporte a múltiplas árvores (n_estimators)
   - Configurações de profundidade e amostragem personalizáveis

3. Redes Neurais (MLP)
   - Arquitetura configurável (hidden_layer_sizes)
   - Suporte a diferentes funções de ativação
   - Regularização via alpha

4. Support Vector Machines (SVM)
   - Suporte a diferentes kernels
   - Parâmetros C e gamma ajustáveis

## Métricas de Avaliação

Para cada modelo, são calculadas as seguintes métricas:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Coeficiente de Determinação (R²)

## Como Usar

### 1. Configuração dos Modelos

Os modelos são configurados através do arquivo `models_to_train.json`. Exemplo de configuração:

```json
{
    "models": [
        {
            "name": "Decision Tree - Geral",
            "type": "Árvore de Decisão",
            "uf": null,
            "parameters": {
                "max_depth": 7,
                "min_samples_split": 5,
                "min_samples_leaf": 2
            }
        }
    ]
}
```

### 2. Execução do Sistema

O sistema é executado através do script `main.py` (cd <seu_caminho>Projeto_TCC_Jun_25 ; python Scripts/main.py), que oferece as seguintes opções:

1. Executar carga inicial de dados
2. Carregar dados pré-processados
3. Treinar modelo individual
4. Carregar modelo treinado
5. Mostrar dados do modelo carregado
6. Treinar modelos da configuração
7. Gerar gráficos de análise

### 3. Visualização de Resultados

Após o treinamento, o sistema gera automaticamente:
- Gráficos comparativos de R² entre modelos
- Análise de tempo de treinamento
- Visualizações de importância de variáveis

Os gráficos são salvos no diretório `plots/` e incluem:
- `r2_comparison.png`: Comparação de R² entre modelos
- `training_time.png`: Tempo de treinamento por modelo
- `feature_importance_heatmap.png`: Heatmap de importância de variáveis
- Gráficos individuais de importância de variáveis para cada modelo

## Análise de Resultados

O sistema permite duas abordagens de análise:
1. Análise geral (sem filtro por UF)
2. Análise específica por estado (com filtro por UF)

Os resultados são automaticamente salvos e podem ser comparados através das visualizações geradas.

## Próximos Passos

1. Executar análise geral com todos os modelos
2. Identificar estados com maior influência nos resultados
3. Realizar análises específicas para estados selecionados
4. Comparar resultados entre modelos gerais e específicos
5. Documentar conclusões e insights no TCC