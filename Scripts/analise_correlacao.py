import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_data():
    """Carrega os dados principais para análise de correlação"""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'DataSources')
    
    # Carrega dados epidemiológicos com chunking para evitar problemas de memória
    print("Carregando dados epidemiológicos...")
    covid_data = pd.read_csv(
        os.path.join(DATA_DIR, 'caso_full_csv', 'caso_full.csv'),
        chunksize=50000
    )
    
    # Processa chunks e mantém apenas uma amostra representativa
    covid_chunks = []
    chunk_count = 0
    for chunk in covid_data:
        # Pega apenas dados de 2020-2021 para reduzir tamanho
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk_filtered = chunk[
            (chunk['date'] >= '2020-03-01') & 
            (chunk['date'] <= '2021-12-31')
        ]
        covid_chunks.append(chunk_filtered)
        chunk_count += 1
        if chunk_count >= 20:  # Limita a 20 chunks
            break
    
    covid_data = pd.concat(covid_chunks, ignore_index=True)
    print(f"Dados COVID carregados: {len(covid_data)} registros")
    
    # Carrega dados de votação 2016
    print("Carregando dados eleitorais...")
    votacao_2016 = pd.read_csv(
        os.path.join(DATA_DIR, 'votacao_candidato', 'votacao_candidato_2016.csv'), 
        encoding="latin1", delimiter=';'
    )
    
    # Normalizar nomes de colunas (importar função do carga_dados)
    from carga_dados import normalize_column_names
    votacao_2016 = normalize_column_names(votacao_2016)
    
    return covid_data, votacao_2016

def prepare_covid_correlation_data(covid_data):
    """Prepara dados COVID para análise de correlação"""
    # Seleciona apenas variáveis numéricas relevantes
    numeric_cols = [
        'epidemiological_week',
        'estimated_population',
        'estimated_population_2019',
        'last_available_confirmed',
        'last_available_confirmed_per_100k_inhabitants',
        'last_available_death_rate',
        'last_available_deaths',
        'order_for_place',
        'new_confirmed',
        'new_deaths'
    ]
    
    # Filtra apenas dados de municípios (não estados)
    city_data = covid_data[covid_data['place_type'] == 'city'].copy()
    
    # Remove valores nulos e infinitos
    correlation_data = city_data[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    return correlation_data

def create_correlation_matrix(data, title, filename):
    """Cria matriz de correlação com visualização"""
    # Calcula matriz de correlação
    correlation_matrix = data.corr()
    
    # Cria figura
    plt.figure(figsize=(12, 10))
    
    # Cria heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix, 
        mask=mask,
        annot=True, 
        cmap='RdBu_r', 
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .8}
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    # Salva figura
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'Docs', 'ResultadosPreliminares')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def analyze_voting_correlations(votacao_data):
    """Analisa correlações nos dados de votação"""
    # Filtra apenas candidatos eleitos
    eleitos = votacao_data[votacao_data['Situação totalização'] == 'Eleito'].copy()
    
    # Cria variáveis numéricas para análise
    eleitos['idade_numerica'] = eleitos['Faixa etária'].map({
        '21 a 24 anos': 22.5, '25 a 29 anos': 27, '30 a 34 anos': 32,
        '35 a 39 anos': 37, '40 a 44 anos': 42, '45 a 49 anos': 47,
        '50 a 54 anos': 52, '55 a 59 anos': 57, '60 a 64 anos': 62,
        '65 a 69 anos': 67, '70 a 74 anos': 72, '75 a 79 anos': 77
    })
    
    # Codifica variáveis categóricas
    eleitos['genero_num'] = eleitos['Gênero'].map({'Masculino': 1, 'Feminino': 0})
    eleitos['educacao_num'] = eleitos['Grau de instrução'].map({
        'Ensino Fundamental incompleto': 1,
        'Ensino Fundamental completo': 2,
        'Ensino Médio incompleto': 3,
        'Ensino Médio completo': 4,
        'Superior incompleto': 5,
        'Superior completo': 6
    })
    
    # Seleciona variáveis para correlação
    voting_numeric = eleitos[['idade_numerica', 'genero_num', 'educacao_num', 'Votos nominais']].dropna()
    
    return voting_numeric

def generate_descriptive_statistics(covid_data, votacao_data):
    """Gera estatísticas descritivas dos dados"""
    print("=== ANÁLISE DESCRITIVA DOS DADOS ===\n")
    
    # Estatísticas COVID
    print("DADOS EPIDEMIOLÓGICOS (COVID-19)")
    print("-" * 40)
    
    city_data = covid_data[covid_data['place_type'] == 'city']
    numeric_cols = ['estimated_population', 'last_available_confirmed', 
                   'last_available_deaths', 'new_confirmed', 'new_deaths']
    
    covid_stats = city_data[numeric_cols].describe()
    print(covid_stats.round(2))
    
    print(f"\nPeríodo dos dados: {covid_data['date'].min()} a {covid_data['date'].max()}")
    print(f"Total de municípios: {city_data['city'].nunique()}")
    print(f"Total de registros: {len(city_data)}")
    
    # Estatísticas Votação
    print("\n\nDADOS ELEITORAIS (2016)")
    print("-" * 40)
    
    eleitos = votacao_data[votacao_data['Situação totalização'] == 'Eleito']
    
    print(f"Total de prefeitos eleitos: {len(eleitos)}")
    print(f"Municípios com dados eleitorais: {eleitos['Município'].nunique()}")
    
    # Distribuição por gênero
    genero_dist = eleitos['Gênero'].value_counts(normalize=True) * 100
    print(f"\nDistribuição por Gênero:")
    for genero, pct in genero_dist.items():
        print(f"  {genero}: {pct:.1f}%")
    
    # Distribuição por educação
    edu_dist = eleitos['Grau de instrução'].value_counts(normalize=True) * 100
    print(f"\nDistribuição por Escolaridade:")
    for edu, pct in edu_dist.items():
        print(f"  {edu}: {pct:.1f}%")
    
    return covid_stats, eleitos

def main():
    """Função principal"""
    print("Iniciando análise de correlação e estatísticas descritivas...")
    
    # Carrega dados
    covid_data, votacao_data = load_data()
    
    # Prepara dados COVID para correlação
    covid_corr_data = prepare_covid_correlation_data(covid_data)
    
    # Cria matriz de correlação COVID
    print("Gerando matriz de correlação - Dados Epidemiológicos...")
    covid_correlation = create_correlation_matrix(
        covid_corr_data, 
        'Matriz de Correlação - Variáveis Epidemiológicas COVID-19',
        'matriz_correlacao_covid.png'
    )
    
    # Analisa correlações votação
    voting_corr_data = analyze_voting_correlations(votacao_data)
    
    # Cria matriz de correlação votação
    print("Gerando matriz de correlação - Dados Eleitorais...")
    voting_correlation = create_correlation_matrix(
        voting_corr_data,
        'Matriz de Correlação - Características dos Prefeitos Eleitos (2016)',
        'matriz_correlacao_eleicoes.png'
    )
    
    # Gera estatísticas descritivas
    print("\nGerando estatísticas descritivas...")
    covid_stats, eleitos_stats = generate_descriptive_statistics(covid_data, votacao_data)
    
    # Salva estatísticas em arquivo
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'Docs', 'ResultadosPreliminares')
    
    with open(os.path.join(OUTPUT_DIR, 'estatisticas_descritivas.txt'), 'w', encoding='utf-8') as f:
        f.write("ANÁLISE DESCRITIVA DOS DADOS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        
        f.write("DADOS EPIDEMIOLÓGICOS (COVID-19)\n")
        f.write("-" * 40 + "\n")
        f.write(str(covid_stats.round(2)))
        f.write(f"\n\nPeríodo: {covid_data['date'].min()} a {covid_data['date'].max()}")
        f.write(f"\nTotal de municípios: {covid_data[covid_data['place_type'] == 'city']['city'].nunique()}")
        
        f.write("\n\nDADOS ELEITORAIS (2016)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total de prefeitos eleitos: {len(eleitos_stats)}\n")
        
        genero_dist = eleitos_stats['Gênero'].value_counts(normalize=True) * 100
        f.write("\nDistribuição por Gênero:\n")
        for genero, pct in genero_dist.items():
            f.write(f"  {genero}: {pct:.1f}%\n")
    
    print(f"\nAnálise concluída! Arquivos salvos em: {OUTPUT_DIR}")
    print("- matriz_correlacao_covid.png")
    print("- matriz_correlacao_eleicoes.png") 
    print("- estatisticas_descritivas.txt")

if __name__ == "__main__":
    main()
