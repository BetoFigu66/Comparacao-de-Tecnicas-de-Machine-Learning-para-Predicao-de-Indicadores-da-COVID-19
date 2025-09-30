"""
Explorador de dados de PIB municipal para integração com modelos COVID-19
Analisa estrutura e variáveis relevantes do arquivo PIB dos Municípios 2010-2021
"""

import pandas as pd
import numpy as np
import os
import sys

def explore_pib_data():
    """Explora a estrutura dos dados de PIB municipal"""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Tentar primeiro o arquivo Excel original
    pib_file_xlsx = os.path.join(project_root, 'DataSources', 'IBGE_Economicos', 'PIB_municipios_2010_2021.xls')
    pib_file_csv = os.path.join(project_root, 'DataSources', 'IBGE_Economicos', 'PIB_municipios_2010_2021.csv')
    
    print("=== EXPLORACAO DOS DADOS DE PIB MUNICIPAL ===")
    
    df_pib = None
    
    # Tentar carregar Excel primeiro
    if os.path.exists(pib_file_xlsx):
        try:
            print(f"Tentando carregar arquivo Excel: {pib_file_xlsx}")
            df_pib = pd.read_excel(pib_file_xlsx)
            print(f"Arquivo Excel carregado com sucesso!")
            print(f"Dataset shape: {df_pib.shape}")
            if 'Ano' in df_pib.columns:
                print(f"Período: {df_pib['Ano'].min()} - {df_pib['Ano'].max()}")
        except Exception as e:
            print(f"Erro ao carregar Excel: {e}")
    
    # Se Excel falhou, tentar CSV com diferentes separadores e encodings
    if df_pib is None:
        print(f"Tentando carregar arquivo CSV: {pib_file_csv}")
        separators = [';', ',', '\t']
        encodings = ['latin1', 'cp1252', 'iso-8859-1', 'utf-8']
        
        for sep in separators:
            for encoding in encodings:
                try:
                    df_pib = pd.read_csv(pib_file_csv, sep=sep, encoding=encoding, on_bad_lines='skip')
                    print(f"CSV carregado com sep='{sep}' e encoding='{encoding}'")
                    print(f"Dataset shape: {df_pib.shape}")
                    if 'Ano' in df_pib.columns:
                        print(f"Período: {df_pib['Ano'].min()} - {df_pib['Ano'].max()}")
                    break
                except Exception as e:
                    continue
            if df_pib is not None:
                break
    
    if df_pib is None:
        print("Erro: Não foi possível carregar nenhum dos arquivos PIB")
        return None
    
    # Analisar colunas
    print(f"\nColunas disponíveis ({len(df_pib.columns)}):")
    for i, col in enumerate(df_pib.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Focar no ano 2020 (ano da pandemia)
    df_2020 = df_pib[df_pib['Ano'] == 2020].copy()
    print(f"\nDados para 2020: {df_2020.shape[0]} municípios")
    
    if df_2020.empty:
        print("Dados de 2020 não disponíveis. Usando 2019...")
        df_2020 = df_pib[df_pib['Ano'] == 2019].copy()
        print(f"Dados para 2019: {df_2020.shape[0]} municípios")
    
    # Identificar colunas de PIB relevantes
    pib_columns = [col for col in df_pib.columns if 'PIB' in col or 'Produto Interno Bruto' in col]
    print(f"\nColunas de PIB identificadas:")
    for col in pib_columns:
        print(f"  - {col}")
    
    # Identificar colunas de atividades econômicas
    atividade_columns = [col for col in df_pib.columns if 'Atividade' in col]
    print(f"\nColunas de atividades econômicas:")
    for col in atividade_columns:
        print(f"  - {col}")
    
    # Analisar dados de 2020/2019
    print(f"\n=== ANALISE DOS DADOS DE {df_2020['Ano'].iloc[0]} ===")
    
    # PIB per capita
    pib_per_capita_col = None
    for col in df_2020.columns:
        if 'per capita' in col.lower():
            pib_per_capita_col = col
            break
    
    if pib_per_capita_col:
        print(f"\nPIB per capita ({pib_per_capita_col}):")
        # Converter para numérico diretamente, já que pode ser float
        pib_per_capita = pd.to_numeric(df_2020[pib_per_capita_col], errors='coerce')
        print(f"  Média: R$ {pib_per_capita.mean():.2f}")
        print(f"  Mediana: R$ {pib_per_capita.median():.2f}")
        print(f"  Min: R$ {pib_per_capita.min():.2f}")
        print(f"  Max: R$ {pib_per_capita.max():.2f}")
        print(f"  Valores nulos: {pib_per_capita.isnull().sum()}")
    
    # PIB total
    pib_total_col = None
    for col in df_2020.columns:
        if 'Produto Interno Bruto' in col and 'per capita' not in col and 'R$ 1.000' in col:
            pib_total_col = col
            break
    
    if pib_total_col:
        print(f"\nPIB total ({pib_total_col}):")
        pib_total = pd.to_numeric(df_2020[pib_total_col], errors='coerce')
        print(f"  Média: R$ {pib_total.mean()/1000:.0f} milhões")
        print(f"  Mediana: R$ {pib_total.median()/1000:.0f} milhões")
        print(f"  Min: R$ {pib_total.min()/1000:.0f} milhões")
        print(f"  Max: R$ {pib_total.max()/1000:.0f} milhões")
        print(f"  Valores nulos: {pib_total.isnull().sum()}")
    
    # Atividades econômicas predominantes
    for i, col in enumerate(atividade_columns):
        print(f"\n{col}:")
        atividades = df_2020[col].value_counts().head(10)
        print("  Top 10 atividades:")
        for atividade, count in atividades.items():
            print(f"    {atividade}: {count} municípios")
    
    # Verificar códigos de município para integração
    print(f"\nCódigos de município:")
    print(f"  Coluna: 'Código do Município'")
    print(f"  Exemplo: {df_2020['Código do Município'].iloc[0]}")
    print(f"  Únicos: {df_2020['Código do Município'].nunique()}")
    
    print(f"\nNomes de município:")
    print(f"  Coluna: 'Nome do Município'")
    print(f"  Exemplos: {list(df_2020['Nome do Município'].head())}")
    print(f"  Únicos: {df_2020['Nome do Município'].nunique()}")
    
    print(f"\nUFs:")
    print(f"  Coluna: 'Sigla da Unidade da Federação'")
    print(f"  Valores: {sorted(df_2020['Sigla da Unidade da Federação'].unique())}")
    
    return df_2020

def create_pib_integration_dataset():
    """Cria dataset de PIB para integração com dados COVID-19"""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Tentar carregar dados usando a mesma lógica da função anterior
    pib_file_xlsx = os.path.join(project_root, 'DataSources', 'IBGE_Economicos', 'PIB_municipios_2010_2021.xls')
    pib_file_csv = os.path.join(project_root, 'DataSources', 'IBGE_Economicos', 'PIB_municipios_2010_2021.csv')
    
    print(f"Procurando arquivo PIB Excel: {pib_file_xlsx}")
    print(f"Arquivo existe: {os.path.exists(pib_file_xlsx)}")
    
    df_pib = None
    
    # Tentar Excel primeiro
    if os.path.exists(pib_file_xlsx):
        try:
            print("Carregando arquivo Excel...")
            df_pib = pd.read_excel(pib_file_xlsx)
            print(f"Excel carregado com sucesso! Shape: {df_pib.shape}")
        except Exception as e:
            print(f"Erro ao carregar Excel: {e}")
            pass
    
    # Se Excel falhou, tentar CSV
    if df_pib is None:
        separators = [';', ',', '\t']
        encodings = ['latin1', 'cp1252', 'iso-8859-1', 'utf-8']
        
        for sep in separators:
            for encoding in encodings:
                try:
                    df_pib = pd.read_csv(pib_file_csv, sep=sep, encoding=encoding, on_bad_lines='skip')
                    break
                except:
                    continue
            if df_pib is not None:
                break
    
    # Verificar se conseguiu carregar os dados
    if df_pib is None:
        print("ERRO: Não foi possível carregar os dados de PIB")
        print(f"Arquivos procurados:")
        print(f"  Excel: {pib_file_xlsx} (existe: {os.path.exists(pib_file_xlsx)})")
        print(f"  CSV: {pib_file_csv} (existe: {os.path.exists(pib_file_csv)})")
        return None
    
    # Usar dados de 2020, se não disponível usar 2019
    if 2020 in df_pib['Ano'].values:
        df_target = df_pib[df_pib['Ano'] == 2020].copy()
        year_used = 2020
    else:
        df_target = df_pib[df_pib['Ano'] == 2019].copy()
        year_used = 2019
    
    print(f"\nCriando dataset de integração com dados de {year_used}")
    print(f"Municípios: {len(df_target)}")
    
    # Selecionar colunas relevantes
    columns_to_keep = [
        'Código do Município',
        'Nome do Município', 
        'Sigla da Unidade da Federação'
    ]
    
    # Adicionar colunas de PIB
    pib_columns = []
    for col in df_target.columns:
        if 'Produto Interno Bruto' in col and 'per capita' in col:
            columns_to_keep.append(col)
            pib_columns.append(col)
        elif 'Produto Interno Bruto' in col and 'R$ 1.000' in col and 'per capita' not in col:
            columns_to_keep.append(col)
            pib_columns.append(col)
    
    # Adicionar colunas de atividades econômicas
    atividade_columns = [col for col in df_target.columns if 'Atividade' in col]
    columns_to_keep.extend(atividade_columns)
    
    # Criar dataset final
    df_integration = df_target[columns_to_keep].copy()
    
    # Renomear colunas para facilitar uso
    rename_dict = {}
    for col in df_integration.columns:
        if 'per capita' in col:
            rename_dict[col] = 'PIB_per_capita'
        elif 'Produto Interno Bruto' in col and 'R$ 1.000' in col:
            rename_dict[col] = 'PIB_total'
        elif col == 'Código do Município':
            rename_dict[col] = 'cod_municipio'
        elif col == 'Nome do Município':
            rename_dict[col] = 'municipio'
        elif col == 'Sigla da Unidade da Federação':
            rename_dict[col] = 'uf'
        elif 'Atividade com maior valor' in col:
            rename_dict[col] = 'atividade_principal'
        elif 'Atividade com segundo maior' in col:
            rename_dict[col] = 'atividade_secundaria'
        elif 'Atividade com terceiro maior' in col:
            rename_dict[col] = 'atividade_terciaria'
    
    df_integration = df_integration.rename(columns=rename_dict)
    
    # Converter PIB para numérico
    if 'PIB_per_capita' in df_integration.columns:
        df_integration['PIB_per_capita'] = pd.to_numeric(df_integration['PIB_per_capita'], errors='coerce')
    
    if 'PIB_total' in df_integration.columns:
        df_integration['PIB_total'] = pd.to_numeric(df_integration['PIB_total'], errors='coerce')
    
    # Salvar dataset de integração
    output_file = os.path.join(project_root, 'Data', 'Processed', 'pib_municipal_integration.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_integration.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nDataset de integração salvo: {output_file}")
    print(f"Shape: {df_integration.shape}")
    print(f"Colunas: {list(df_integration.columns)}")
    
    # Estatísticas finais
    print(f"\nEstatísticas do dataset de integração:")
    if 'PIB_per_capita' in df_integration.columns:
        print(f"  PIB per capita - Média: R$ {df_integration['PIB_per_capita'].mean():.2f}")
        print(f"  PIB per capita - Nulos: {df_integration['PIB_per_capita'].isnull().sum()}")
    
    if 'PIB_total' in df_integration.columns:
        print(f"  PIB total - Média: R$ {df_integration['PIB_total'].mean()/1000:.0f} milhões")
        print(f"  PIB total - Nulos: {df_integration['PIB_total'].isnull().sum()}")
    
    if 'atividade_principal' in df_integration.columns:
        print(f"  Atividades principais únicas: {df_integration['atividade_principal'].nunique()}")
        print(f"  Top 5 atividades principais:")
        for atividade, count in df_integration['atividade_principal'].value_counts().head().items():
            print(f"    {atividade}: {count}")
    
    return df_integration

if __name__ == "__main__":
    # Explorar dados
    df_2020 = explore_pib_data()
    
    if df_2020 is not None:
        # Criar dataset de integração
        df_integration = create_pib_integration_dataset()
