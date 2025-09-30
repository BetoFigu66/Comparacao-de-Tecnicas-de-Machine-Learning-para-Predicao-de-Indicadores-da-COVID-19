# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:05:00 2025

@author: betof
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import unicodedata
import re

use_death_rate = False

def normalize_variable_name(name):
    """
    Normaliza nomes de variáveis removendo acentos e substituindo espaços por underscores
    """
    if not isinstance(name, str):
        return str(name)
    
    # Remover acentos
    normalized = unicodedata.normalize('NFD', name)
    without_accents = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Substituir espaços por underscores
    with_underscores = without_accents.replace(' ', '_')
    
    # Remover caracteres especiais (manter apenas letras, números e underscores)
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', with_underscores)
    
    # Remover underscores múltiplos
    clean_name = re.sub(r'_+', '_', clean_name)
    
    # Remover underscores no início e fim
    clean_name = clean_name.strip('_')
    
    return clean_name

def remove_duplicate_columns(df, logger=None):
    """
    Remove colunas duplicadas com sufixos _x, _y mantendo apenas uma versão limpa
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Iniciando limpeza de colunas duplicadas (_x, _y)...")
    
    # Encontrar todas as colunas com _x
    x_columns = [col for col in df.columns if col.endswith('_x')]
    
    columns_to_drop = []
    columns_to_rename = {}
    duplicates_found = 0
    
    for x_col in x_columns:
        # Obter nome base (sem _x)
        base_name = x_col[:-2]  # Remove '_x'
        y_col = base_name + '_y'
        
        # Verificar se existe a coluna correspondente _y
        if y_col in df.columns:
            logger.info(f"Analisando par duplicado: {x_col} e {y_col}")
            
            # Verificar se os valores são idênticos
            x_values = df[x_col].fillna('__NULL__')  # Tratar NaN para comparação
            y_values = df[y_col].fillna('__NULL__')
            
            if x_values.equals(y_values):
                # Valores idênticos - manter apenas uma versão com nome limpo
                columns_to_rename[x_col] = base_name
                columns_to_drop.append(y_col)
                duplicates_found += 1
                logger.info(f"✅ Valores idênticos - mantendo como '{base_name}', removendo '{y_col}'")
            else:
                # Valores diferentes - verificar qual tem mais dados válidos
                x_valid = df[x_col].notna().sum()
                y_valid = df[y_col].notna().sum()
                
                if x_valid >= y_valid:
                    columns_to_rename[x_col] = base_name
                    columns_to_drop.append(y_col)
                    logger.info(f"⚖️ Valores diferentes - mantendo '{x_col}' ({x_valid} válidos) como '{base_name}', removendo '{y_col}' ({y_valid} válidos)")
                else:
                    columns_to_rename[y_col] = base_name
                    columns_to_drop.append(x_col)
                    logger.info(f"⚖️ Valores diferentes - mantendo '{y_col}' ({y_valid} válidos) como '{base_name}', removendo '{x_col}' ({x_valid} válidos)")
                
                duplicates_found += 1
    
    # Aplicar as mudanças
    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)
        logger.info(f"Renomeadas {len(columns_to_rename)} colunas")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Removidas {len(columns_to_drop)} colunas duplicadas")
    
    logger.info(f"✅ Limpeza concluída: {duplicates_found} pares de duplicatas processados")
    
    return df

def setup_data_logging():
    """Configura o sistema de logging para carga de dados"""
    # Criar diretório de logs se não existir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    log_filename = f"carga_dados_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()  # Também exibe no console
        ]
    )
    
    # Obter logger específico para carga de dados
    logger = logging.getLogger('carga_dados')
    logger.info(f"Sistema de logging para carga de dados iniciado. Arquivo: {log_filepath}")
    
    return logger

# Inicializar logger global
data_logger = setup_data_logging()

def process_ibge_data(data_dir, processed_data_dir):
    """Processa dados do IBGE e cria arquivo ibge_covid_variables.csv"""
    
    ibge_file = os.path.join(data_dir, 'IBGE', 'Base_MUNIC_2020.xlsx')
    output_file = os.path.join(processed_data_dir, 'ibge_covid_variables.csv')
    
    if not os.path.exists(ibge_file):
        data_logger.warning(f"Arquivo IBGE não encontrado: {ibge_file}")
        return False
    
    data_logger.info(f"Processando dados IBGE de: {ibge_file}")
    
    try:
        # Carregar dados do IBGE
        xl = pd.ExcelFile(ibge_file)
        data_logger.info(f"Sheets disponíveis no IBGE: {xl.sheet_names}")
        
        # Variáveis selecionadas baseadas na relevância para COVID-19
        # Focar nas sheets mais importantes que existem no arquivo
        selected_sheets = []
        
        # Verificar quais sheets existem e são relevantes (nomes exatos do arquivo)
        relevant_sheets = ['COVID-19', 'Recursos humanos', 'Habita�o', 'Meio ambiente', 'Gest�o de riscos', 'Vari�veis externa']
        
        for sheet in relevant_sheets:
            if sheet in xl.sheet_names:
                selected_sheets.append(sheet)
                data_logger.info(f"Sheet selecionada: {sheet}")
        
        combined_data = None
        
        # Processar cada sheet selecionada
        for sheet_name in selected_sheets:
            try:
                data_logger.info(f"Processando sheet: {sheet_name}")
                df = pd.read_excel(ibge_file, sheet_name=sheet_name)
                
                # Encontrar coluna de código do município
                mun_col = None
                for col in df.columns:
                    if 'CodMun' in str(col) or ('Cod' in str(col) and 'Mun' in str(col)):
                        mun_col = col
                        break
                
                if mun_col is None:
                    data_logger.warning(f"Coluna de código do município não encontrada em {sheet_name}")
                    continue
                
                # Selecionar colunas relevantes (numéricas E categóricas)
                exclude_cols = [mun_col, 'UF', 'Cod UF', 'Mun', 'PopMun', 'Faixa_pop', 'Regiao']
                
                relevant_cols = [mun_col]
                for col in df.columns:
                    if col not in exclude_cols and not pd.isna(df[col]).all():
                        # Verificar se a coluna tem dados úteis
                        unique_vals = df[col].nunique()
                        if unique_vals > 1 and unique_vals < len(df) * 0.9:  # Não constante e não muito única
                            relevant_cols.append(col)
                
                if len(relevant_cols) > 1:  # Se temos mais que só código do município
                    sheet_data = df[relevant_cols].copy()
                    
                    # Renomear colunas (exceto código do município)
                    rename_dict = {}
                    for col in relevant_cols:
                        if col != mun_col:
                            rename_dict[col] = f'IBGE_{sheet_name.replace(" ", "_").replace("-", "_")}_{col}'
                    
                    sheet_data = sheet_data.rename(columns=rename_dict)
                    sheet_data = sheet_data.rename(columns={mun_col: 'CodMun'})
                    
                    # Converter colunas categóricas em dummies e limpar tipos mistos
                    categorical_cols = sheet_data.select_dtypes(include=['object']).columns.tolist()
                    if 'CodMun' in categorical_cols:
                        categorical_cols.remove('CodMun')
                    
                    for cat_col in categorical_cols:
                        # Padronizar valores de dados omitidos/ausentes
                        # Primeiro tratar valores NaN reais (antes de converter para string)
                        sheet_data[cat_col] = sheet_data[cat_col].fillna('Omitido')
                        
                        # Depois converter para string para evitar tipos mistos
                        sheet_data[cat_col] = sheet_data[cat_col].astype(str)
                        
                        # Tratar casos onde NaN virou string "nan" durante astype(str)
                        sheet_data[cat_col] = sheet_data[cat_col].replace('nan', 'Omitido')
                        
                        # Tratamento específico para campo Mcov01 (isolamento social)
                        if 'Mcov01' in cat_col:
                            mcov01_mappings = {
                                'Sim, a população foi orientada para permanecer em isolamento social': 'orientacao_isolamento',
                                'Sim, foi decretado isolamento social': 'decretado_isolamento',
                                'Recusa': 'Omitido',
                                'Não': 'nao_ao_isolamento'
                            }
                            for original_val, new_val in mcov01_mappings.items():
                                # Normalizar o valor mapeado
                                normalized_val = normalize_variable_name(new_val)
                                sheet_data[cat_col] = sheet_data[cat_col].replace(original_val, normalized_val)
                                if original_val in sheet_data[cat_col].values:
                                    data_logger.info(f"Mapeando Mcov01: '{original_val}' -> '{normalized_val}'")
                        
                        # Tratamento específico para campo Mmam01 (órgão gestor meio ambiente)
                        if 'Mmam01' in cat_col:
                            mmam01_mappings = {
                                'Não possui estrutura': 'nao_possui_orgao_gestor',
                                'Órgão da administração indireta': 'administracao_indireta_orgao_gestor',
                                'Secretaria em conjunto com outras políticas setoriais': 'em_conjunto_orgao_gestor',
                                'Secretaria exclusiva': 'exclusivo_orgao_gestor',
                                'Setor subordinado a outra secretaria': 'subordinado_outro_orgao_gestor',
                                'Setor subordinado diretamente à chefia do Executivo': 'subordinado_executivo_orgao_gestor'
                            }
                            for original_val, new_val in mmam01_mappings.items():
                                # Normalizar o valor mapeado
                                normalized_val = normalize_variable_name(new_val)
                                sheet_data[cat_col] = sheet_data[cat_col].replace(original_val, normalized_val)
                                if original_val in sheet_data[cat_col].values:
                                    data_logger.info(f"Mapeando Mmam01: '{original_val}' -> '{normalized_val}'")
                        
                        # Tratar strings representando valores ausentes (incluindo Recusa para outros campos)
                        missing_values = ['-', 'nan', 'None', 'Não informou', 'Não informado', 'NaN', 'null', 'NULL', '', 'Recusa']
                        for missing_val in missing_values:
                            sheet_data[cat_col] = sheet_data[cat_col].replace(missing_val, 'Omitido')
                        
                        # Normalizar todos os valores categóricos (remover acentos e espaços)
                        unique_vals = sheet_data[cat_col].unique()
                        for val in unique_vals:
                            if isinstance(val, str) and val not in ['Omitido']:  # Não normalizar 'Omitido'
                                normalized_val = normalize_variable_name(val)
                                if normalized_val != val:  # Só substituir se houve mudança
                                    sheet_data[cat_col] = sheet_data[cat_col].replace(val, normalized_val)
                                    data_logger.info(f"Normalizando valor em {cat_col}: '{val}' -> '{normalized_val}'")
                        
                        # Truncar nomes de valores ainda muito longos (>50 caracteres) - após mapeamentos
                        unique_vals = sheet_data[cat_col].unique()
                        for val in unique_vals:
                            if isinstance(val, str) and len(val) > 50:
                                truncated_val = val[:47] + "..."
                                sheet_data[cat_col] = sheet_data[cat_col].replace(val, truncated_val)
                                data_logger.info(f"Truncando valor longo em {cat_col}: '{val}' -> '{truncated_val}'")
                        
                        data_logger.info(f"Padronizando valores omitidos em {cat_col}")
                        
                        if sheet_data[cat_col].nunique() <= 10:  # Só para colunas com poucas categorias
                            # Não usar dummy_na=True pois já tratamos todos os NaN como 'Omitido'
                            dummies = pd.get_dummies(sheet_data[cat_col], prefix=cat_col, dummy_na=False, dtype=int)
                            
                            # Normalizar nomes das colunas dummy criadas
                            normalized_columns = {}
                            for col in dummies.columns:
                                normalized_col = normalize_variable_name(col)
                                if normalized_col != col:
                                    normalized_columns[col] = normalized_col
                            
                            if normalized_columns:
                                dummies = dummies.rename(columns=normalized_columns)
                                data_logger.info(f"Normalizando {len(normalized_columns)} nomes de colunas dummy para {cat_col}")
                            
                            sheet_data = pd.concat([sheet_data, dummies], axis=1)
                            sheet_data = sheet_data.drop(columns=[cat_col])
                        else:
                            # Para colunas com muitas categorias, remover para evitar problemas
                            data_logger.warning(f"Removendo coluna {cat_col} (muitas categorias: {sheet_data[cat_col].nunique()})")
                            sheet_data = sheet_data.drop(columns=[cat_col])
                    
                    # Combinar com dados existentes
                    if combined_data is None:
                        combined_data = sheet_data
                    else:
                        # Verificar colunas duplicadas antes do merge
                        overlapping_cols = set(combined_data.columns) & set(sheet_data.columns)
                        overlapping_cols.discard('CodMun')  # Remover a chave de join
                        
                        if overlapping_cols:
                            data_logger.warning(f"Colunas duplicadas detectadas em {sheet_name}: {overlapping_cols}")
                            # Renomear colunas duplicadas no sheet_data para evitar _x, _y
                            rename_dict = {}
                            for col in overlapping_cols:
                                new_name = f"{col}_{sheet_name}"
                                rename_dict[col] = new_name
                                data_logger.info(f"Renomeando coluna duplicada: {col} -> {new_name}")
                            sheet_data = sheet_data.rename(columns=rename_dict)
                        
                        combined_data = combined_data.merge(sheet_data, on='CodMun', how='outer')
                    
                    data_logger.info(f"Adicionadas {len(relevant_cols)-1} variáveis de {sheet_name}")
                
            except Exception as e:
                data_logger.error(f"Erro ao processar sheet {sheet_name}: {e}")
                continue
        
        if combined_data is None or combined_data.empty:
            data_logger.error("Nenhum dado IBGE foi processado com sucesso")
            return False
        
        # Limpar dados IBGE...
        data_logger.info("Limpando dados IBGE...")
        
        # Aplicar limpeza inteligente de colunas duplicadas (_x, _y) no IBGE
        combined_data = remove_duplicate_columns(combined_data, data_logger)
        
        # Remover linhas onde CodMun é nulo
        combined_data = combined_data.dropna(subset=['CodMun'])
        
        # Converter CodMun para int
        combined_data['CodMun'] = combined_data['CodMun'].astype(int)
        
        # Tratar valores missing - substituir por 0 onde faz sentido
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'CodMun':
                combined_data[col] = combined_data[col].fillna(0)
        
        # Salvar dados processados
        combined_data.to_csv(output_file, index=False)
        data_logger.info(f"Dados IBGE salvos em: {output_file}")
        data_logger.info(f"Shape final IBGE: {combined_data.shape}")
        data_logger.info(f"Variáveis IBGE criadas: {len([col for col in combined_data.columns if col.startswith('IBGE_')])}")
        
        return True
        
    except Exception as e:
        data_logger.error(f"Erro geral no processamento IBGE: {e}")
        return False

def process_pib_data(data_dir, processed_data_dir):
    """Processa dados do PIB e cria arquivo pib_covid_variables.csv"""
    
    pib_file = os.path.join(data_dir, 'IBGE_Economicos', 'PIB_municipios_2010_2021.xls')
    output_file = os.path.join(processed_data_dir, 'pib_covid_variables.csv')
    
    if not os.path.exists(pib_file):
        data_logger.warning(f"Arquivo PIB não encontrado: {pib_file}")
        return False
    
    data_logger.info(f"Processando dados PIB de: {pib_file}")
    
    try:
        # Carregar dados do PIB
        df_pib = pd.read_excel(pib_file)
        data_logger.info(f"Dados PIB carregados: {df_pib.shape}")
        data_logger.info(f"Colunas PIB: {df_pib.columns.tolist()}")
        
        # Filtrar para o ano 2020 (mais próximo dos dados de COVID)
        if 'Ano' in df_pib.columns:
            df_pib_2020 = df_pib[df_pib['Ano'] == 2020].copy()
            if df_pib_2020.empty:
                # Se não tem 2020, usar o ano mais recente disponível
                latest_year = df_pib['Ano'].max()
                df_pib_2020 = df_pib[df_pib['Ano'] == latest_year].copy()
                data_logger.info(f"Ano 2020 não encontrado, usando {latest_year}")
        else:
            df_pib_2020 = df_pib.copy()
        
        data_logger.info(f"Dados PIB filtrados: {df_pib_2020.shape}")
        
        # Encontrar coluna de código do município
        mun_col = None
        for col in df_pib_2020.columns:
            if 'código' in str(col).lower() and 'município' in str(col).lower():
                mun_col = col
                break
        
        if mun_col is None:
            # Tentar outras variações
            for col in df_pib_2020.columns:
                if 'cod' in str(col).lower() and 'mun' in str(col).lower():
                    mun_col = col
                    break
        
        if mun_col is None:
            data_logger.error("Coluna de código do município não encontrada nos dados PIB")
            return False
        
        # Selecionar colunas relevantes
        relevant_cols = [mun_col]
        
        # Adicionar colunas de PIB e indicadores econômicos
        for col in df_pib_2020.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['pib', 'valor', 'adicionado', 'impostos', 'per capita']):
                if col != mun_col and df_pib_2020[col].dtype in ['int64', 'float64']:
                    relevant_cols.append(col)
        
        if len(relevant_cols) <= 1:
            data_logger.warning("Nenhuma coluna PIB relevante encontrada")
            return False
        
        # Criar dataset final
        pib_data = df_pib_2020[relevant_cols].copy()
        
        # Renomear colunas
        rename_dict = {mun_col: 'CodMun'}
        for col in relevant_cols:
            if col != mun_col:
                rename_dict[col] = f'PIB_{col.replace(" ", "_").replace("-", "_")}'
        
        pib_data = pib_data.rename(columns=rename_dict)
        
        # Limpar dados
        pib_data = pib_data.dropna(subset=['CodMun'])
        pib_data['CodMun'] = pib_data['CodMun'].astype(int)
        
        # Tratar valores missing
        numeric_cols = pib_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'CodMun':
                pib_data[col] = pib_data[col].fillna(0)
        
        # Salvar dados processados
        pib_data.to_csv(output_file, index=False)
        data_logger.info(f"Dados PIB salvos em: {output_file}")
        data_logger.info(f"Shape final PIB: {pib_data.shape}")
        data_logger.info(f"Variáveis PIB criadas: {len([col for col in pib_data.columns if col.startswith('PIB_')])}")
        
        return True
        
    except Exception as e:
        data_logger.error(f"Erro geral no processamento PIB: {e}")
        return False

def normalize_column_names(df):
    """
    Normaliza nomes de colunas para tratar variações entre diferentes fontes de dados eleitorais.
    
    Mapeamentos suportados:
    - 'Sigla partido' -> 'Partido'
    - 'Situação de totalização' -> 'Situação totalização'
    """
    # Dicionário de mapeamento de nomes alternativos para nomes padrão
    column_mappings = {
        'Sigla partido': 'Partido',
        'Situação de totalização': 'Situação totalização'
    }
    
    # Aplicar mapeamentos se as colunas existirem
    df_normalized = df.copy()
    for old_name, new_name in column_mappings.items():
        if old_name in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={old_name: new_name})
            data_logger.info(f"Coluna renomeada: '{old_name}' -> '{new_name}'")
    
    return df_normalized

def safe_drop_columns(df, columns_to_drop):
    """
    Remove colunas de forma segura, ignorando colunas que não existem.
    
    Args:
        df: DataFrame
        columns_to_drop: Lista de nomes de colunas para remover
    
    Returns:
        DataFrame com as colunas removidas (apenas as que existiam)
    """
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    missing_columns = [col for col in columns_to_drop if col not in df.columns]
    
    if missing_columns:
        data_logger.info(f"Colunas não encontradas (ignoradas): {missing_columns}")
    
    if existing_columns:
        data_logger.info(f"Colunas removidas: {existing_columns}")
        return df.drop(columns=existing_columns)
    else:
        data_logger.info("Nenhuma coluna foi removida (nenhuma encontrada)")
        return df

def safe_rename_columns(df, column_mappings):
    """
    Renomeia colunas de forma segura, ignorando colunas que não existem.
    
    Args:
        df: DataFrame
        column_mappings: Dicionário {nome_antigo: nome_novo}
    
    Returns:
        DataFrame com as colunas renomeadas (apenas as que existiam)
    """
    existing_mappings = {old: new for old, new in column_mappings.items() if old in df.columns}
    missing_columns = [old for old in column_mappings.keys() if old not in df.columns]
    
    if missing_columns:
        data_logger.info(f"Colunas para renomear não encontradas (ignoradas): {missing_columns}")
    
    if existing_mappings:
        data_logger.info(f"Colunas renomeadas: {existing_mappings}")
        return df.rename(columns=existing_mappings)
    else:
        data_logger.info("Nenhuma coluna foi renomeada (nenhuma encontrada)")
        return df

def process_and_save_data():
    data_logger.info("Iniciando processo de carga e tratamento de dados...")
    # Define o diretório base do script para construção de caminhos relativos
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'DataSources')
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'Processed')

    # Garante que o diretório de saída exista
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    ###################################################################################################################
    # Passo 0: Processar dados auxiliares (IBGE e PIB)
    ###################################################################################################################
    data_logger.info("=== PROCESSANDO DADOS AUXILIARES ===")
    
    # Processar dados do IBGE
    data_logger.info("Processando dados do IBGE...")
    ibge_success = process_ibge_data(DATA_DIR, PROCESSED_DATA_DIR)
    if ibge_success:
        data_logger.info("✅ Dados IBGE processados com sucesso")
    else:
        data_logger.warning("⚠️ Falha no processamento dos dados IBGE")
    
    # Processar dados do PIB
    data_logger.info("Processando dados do PIB...")
    pib_success = process_pib_data(DATA_DIR, PROCESSED_DATA_DIR)
    if pib_success:
        data_logger.info("✅ Dados PIB processados com sucesso")
    else:
        data_logger.warning("⚠️ Falha no processamento dos dados PIB")
    
    # Criar arquivo de integração PIB municipal
    data_logger.info("Criando arquivo de integração PIB municipal...")
    try:
        from pib_data_explorer import create_pib_integration_dataset
        create_pib_integration_dataset()
        data_logger.info("✅ Arquivo de integração PIB criado com sucesso")
    except Exception as e:
        data_logger.warning(f"⚠️ Falha na criação do arquivo de integração PIB: {e}")

    ###################################################################################################################
    # Passo 1: Carregar os dados de prefeitos eleitos em 2016, limpando os dados que não serão usados.
    ###################################################################################################################
    # Carregando os dados a serem usados primeiramente, que são os da eleição de 2016
    df_votacao_2016 = pd.read_csv(os.path.join(DATA_DIR, 'votacao_candidato', 'votacao_candidato_2016.csv'), encoding="latin1", delimiter=';')
    
    # Normalizar nomes de colunas para tratar variações entre fontes
    df_votacao_2016 = normalize_column_names(df_votacao_2016)
    
    # Verificar se as colunas essenciais existem
    required_columns = ['Situação totalização', 'Partido', 'Município']
    missing_columns = [col for col in required_columns if col not in df_votacao_2016.columns]
    if missing_columns:
        data_logger.error(f"Colunas obrigatórias não encontradas: {missing_columns}")
        data_logger.error(f"Colunas disponíveis: {list(df_votacao_2016.columns)}")
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
    data_logger.info(f"Dados eleitorais carregados: {df_votacao_2016.shape[0]} registros, {df_votacao_2016.shape[1]} colunas")

    # Padronizando os campo de cidade para maiúsculas pois este será usado como chave para merge dos 2 bancos
    df_votacao_2016['Município'] = df_votacao_2016['Município'].str.upper()

    # Limpar os dados que não serão usados
    # Somente os Eleitos
    df_eleitos_2016 = df_votacao_2016[df_votacao_2016['Situação totalização'] == 'Eleito']
    # df_eleitos_2016.info() 
    # 0   'Ano de eleição'        65337 non-null  int64 
    # 1   Cargo                 65337 non-null  object ## Todos são prefeito
    # 2   Código município      65337 non-null  int64 
    # 3   'Cor/raça'              65337 non-null  object
    # 4   'Estado civil'          65337 non-null  object
    # 5   'Faixa etária'          65337 non-null  object
    # 6   'Gênero'                65337 non-null  object
    # 7   'Grau de instrução'     65337 non-null  object
    # 8   Município             65337 non-null  object
    # 9   Nome candidato        65337 non-null  object
    # 10  Número candidato      65337 non-null  int64 
    # 11  'Ocupação'              65337 non-null  object
    # 12  'Partido'               65337 non-null  object
    # 13  Região                65337 non-null  object
    # 14  Situação totalização  65337 non-null  object ## Filtrados por eleitos
    # 15  'Turno'                 65337 non-null  int64 
    # 16  'UF'                    65337 non-null  object
    # 17  Zona                  65337 non-null  int64 
    # 18  Votos válidos         65337 non-null  int64 
    # 19  Votos nominais        65337 non-null  int64 
    # 20  Data de carga         65337 non-null  object


    # Filtragem dos prefeitos eleitos duplicados para a mesma cidade
    # Lista de colunas desnecessárias para o modelo (podem não existir em todas as fontes)
    columns_to_remove = [
        'Ano de eleição', 'Cargo', 'Código município', 'Nome candidato', 
        'Número candidato', 'Situação totalização', 'Zona',
        'Votos válidos', 'Votos nominais', 'Turno', 'Data de carga'
    ]
    df_eleitos_2016_filtrado = safe_drop_columns(df_eleitos_2016, columns_to_remove)

    # df_eleitos_2016.drop(columns=['Turno'], inplace=True)

    # Renomear colunas para nomes padronizados (de forma segura)
    column_rename_mappings = {
        'Cor/raça': 'cor_raca', 
        'Estado civil': 'estado_civil', 
        'Faixa etária': 'faixa_etaria', 
        'Gênero': 'genero', 
        'Grau de instrução': 'grau_de_instrucao', 
        'Ocupação': 'ocupacao', 
        'Partido': 'partido', 
        'Turno': 'turno', 
        'Região': 'regiao', 
        'UF': 'uf', 
        'Município': 'municipio'
    }
    df_eleitos_2016_filtrado = safe_rename_columns(df_eleitos_2016_filtrado, column_rename_mappings)
    
    # Verificar se as colunas essenciais para o modelo existem após o processamento
    essential_columns = ['municipio', 'uf', 'partido', 'cor_raca', 'estado_civil', 'faixa_etaria', 'genero', 'grau_de_instrucao', 'ocupacao', 'regiao']
    missing_essential = [col for col in essential_columns if col not in df_eleitos_2016_filtrado.columns]
    
    if missing_essential:
        data_logger.error(f"Colunas essenciais para o modelo não encontradas: {missing_essential}")
        data_logger.error(f"Colunas disponíveis após processamento: {list(df_eleitos_2016_filtrado.columns)}")
        raise ValueError(f"Colunas essenciais ausentes: {missing_essential}")
    data_logger.info(f"Todas as colunas essenciais estão presentes: {essential_columns}")
    data_logger.info(f"Colunas finais: {list(df_eleitos_2016_filtrado.columns)}")

    df_eleitos_2016_filtrado = df_eleitos_2016_filtrado.drop_duplicates()
    df_eleitos_2016_filtrado.describe().T

    df_municipios_duplicados = df_eleitos_2016_filtrado.groupby(['municipio', 'uf']).count()
    df_municipios_duplicados = df_municipios_duplicados[df_municipios_duplicados['partido'] != 1]
    df_municipios_duplicados = df_municipios_duplicados.reset_index()
    df_municipios_duplicados = df_municipios_duplicados[['municipio', 'uf']]
    percent = 100.0 * df_municipios_duplicados.shape[0] / df_eleitos_2016_filtrado.shape[0] 
    data_logger.info(f"Descartados por duplicidade {df_municipios_duplicados.shape[0]} sendo {percent:.2f}% do total")

    data_logger.info(f"Número de linhas antes de descartar duplicados: {df_eleitos_2016_filtrado.shape[0]}")
    # df_eleitos_filtrado.columns
    df_eleitos_2016_filtrado_merged = df_eleitos_2016_filtrado.merge(df_municipios_duplicados, on=['municipio', 'uf'], how='outer', indicator='origem')
    # df_eleitos_2016_filtrado_merged['origem'].value_counts()
    df_eleitos_2016_filtrado = df_eleitos_2016_filtrado_merged[df_eleitos_2016_filtrado_merged['origem'] == 'left_only']
    df_eleitos_2016_filtrado = df_eleitos_2016_filtrado.drop(columns=['origem'])
    data_logger.info(f"Número de linhas depois de descartar duplicados: {df_eleitos_2016_filtrado_merged.shape[0]}")

    df_eleitos_2016_filtrado_merged.describe().T

    ### Desta parte acima, o que vai ser usado é apenas o df_eleitos_2016_filtrado_merged
    df_eleitos_2016_filtrado_merged.columns

    # df_eleitos_2016_filtrado_merged.info()
    #  #   Column             Non-Null Count  Dtype 
    # ---  ------             --------------  ----- 
    #  0   cor_raca           5443 non-null   object
    #  1   estado_civil       5443 non-null   object
    #  2   faixa_etaria       5443 non-null   object
    #  3   genero             5443 non-null   object
    #  4   grau_de_instrucao  5443 non-null   object
    #  5   municipio          5443 non-null   object
    #  6   ocupacao           5443 non-null   object
    #  7   partido            5443 non-null   object
    #  8   regiao             5443 non-null   object
    #  9   uf                 5443 non-null   object


    ###################################################################################################################
    # Passo 2: Carregra os dados de casos e mortes por Covid e sumarizar de acordo com a necessidade do TCC:
    # Status: Feito
    #       2.1. Número de casos e mortes por município ao final de 2020 (caso_full_final_2020)
    #       2.2. Número de casos e mortes por município ao final de 2024 (caso_full_final_2024)
    #       2.3. População estimada de cada município
    #       2.4. Número de casos e mortes por habitante por município ao final de 2020
    #       2.5. Número de casos e mortes por habitante por município ao final de 2024
    #       2.6. Eliminar os atributos que não serão usados
    ###################################################################################################################

# Carrega os dados de casos e mortes por Covid, por dia/município
    caso_full = pd.read_csv(os.path.join(DATA_DIR, 'caso_full_csv', 'caso_full.csv'))
    # Padronizando os campo de cidade para maiúsculas pois este será usado como chave para merge dos 2 bancos
    caso_full['city_upper'] = caso_full['city'].str.upper()

 
    caso_full.date.head()
    # caso_full.info()

     # 0   city                                           object 
     # 1   city_ibge_code                                 float64
     # 2   date                                           object  >> Usar como filtro para pegar a última data de 2020
     # 3   epidemiological_week                           int64  
     # 4   estimated_population                           float64 >> Usar para classificar pelo tamanho do município
     # 5   estimated_population_2019                      float64
     # 6   is_last                                        bool   
     # 7   is_repeated                                    bool   
     # 8   last_available_confirmed                       int64  
     # 9   last_available_confirmed_per_100k_inhabitants  float64 >> Usar como Y1
     # 10  last_available_date                            object 
     # 11  last_available_death_rate                      float64
     # 12  last_available_deaths                          int64   >> Usar como Y2
     # 13  order_for_place                                int64  
     # 14  place_type                                     object >> Filtrar city
     # 15  state                                          object >> Usar para o join
     # 16  new_confirmed                                  int64  
     # 17  new_deaths                                     int64  
     # 18  city_upper                                     object >> Usar para o join


    # 1. Filtrar caso_full para place_type = 'city' e date <= '2020-12-31'
    caso_full_filtrado = caso_full[
        (caso_full['place_type'] == 'city') &
        (caso_full['date'] <= '2020-12-31')
    ]

    # 2. Encontrar a linha com a data mais recente para cada combinação de city_upper e state
    caso_full_final_2020 = caso_full_filtrado.sort_values('date', ascending=False).drop_duplicates(
        subset=['city_upper', 'state'], keep='first'
    )

    # 3. Filtrar caso_full para place_type = 'city' e date <= '2020-12-31'
    caso_full_filtrado = caso_full[
        (caso_full['place_type'] == 'city') &
        (caso_full['date'] <= '2024-12-31')
    ]

    # 4. Encontrar a linha com a data mais recente para cada combinação de city_upper e state
    caso_full_final_2024 = caso_full_filtrado.sort_values('date', ascending=False).drop_duplicates(
        subset=['city_upper', 'state'], keep='first'
    )

    caso_full_final_2024.columns
    # ['city', 'city_ibge_code', 'date', 'epidemiological_week',
    #        'estimated_population', 'estimated_population_2019', 'is_last',
    #        'is_repeated', 'last_available_confirmed',
    #        'last_available_confirmed_per_100k_inhabitants', 'last_available_date',
    #        'last_available_death_rate', 'last_available_deaths', 'order_for_place',
    #        'place_type', 'state', 'new_confirmed', 'new_deaths', 'city_upper']

    # IMPORTANTE: Manter city_ibge_code para integração IBGE/PIB
    columns_to_drop = ['city', 'date', 'epidemiological_week',
           'estimated_population_2019', 'is_last', 'is_repeated', 'last_available_confirmed',
           'last_available_date', 'last_available_deaths', 'order_for_place',
           'place_type', 'new_confirmed', 'new_deaths']
    if not use_death_rate:
        caso_full_final_2020 ['death_per_100k_inhabitants'] = caso_full_final_2020 ['last_available_deaths'] / caso_full_final_2020 ['estimated_population_2019']
        caso_full_final_2024 ['death_per_100k_inhabitants'] = caso_full_final_2024 ['last_available_deaths'] / caso_full_final_2024 ['estimated_population_2019']
        columns_to_drop.append('last_available_death_rate')
    caso_full_final_2020 = caso_full_final_2020.drop(columns=columns_to_drop)
    # caso_full_final_2020.columns
    caso_full_final_2020.rename(columns={'city_upper': 'municipio', 'state': 'uf'}, inplace=True)

    caso_full_final_2024 = caso_full_final_2024.drop(columns=columns_to_drop)
    caso_full_final_2024.rename(columns={'city_upper': 'municipio', 'state': 'uf'}, inplace=True)

    # Campos a serem preenchidos e usados:
        # X:
        #     caso_full_final_2020: ['estimated_population', 'uf', 'municipio'] 
        #     df_eleitos_2016_filtrado_merged:  ['cor_raca', 'estado_civil', 'faixa_etaria', 'genero',
        #       'grau_de_instrucao', 'ocupacao', 'partido', 'regiao', 'uf']
        # Y:
        #     'last_available_confirmed_per_100k_inhabitants',
        #     'last_available_death_rate',
        #     'deaths_per_case_rate' = 'last_available_death_rate' / 'last_available_confirmed_per_100k_inhabitants'
        #     

    ###################################################################################################################
    # Passo 3: Fazer o merge das informações dos 2 datasets
    # Status: Feito
    ###################################################################################################################

    df_eleitos_2016_filtrado_merged.shape[0]
    # 5443
    caso_full_final_2020.shape[0]
    # 5589

    df_final_2020 = pd.merge(
        df_eleitos_2016_filtrado_merged,
        caso_full_final_2020,
        on=['municipio', 'uf'],
        how='left'
    )
    
    data_logger.info(f"Dados básicos integrados: {df_final_2020.shape}")
    
    ###################################################################################################################
    # Passo 3.1: Integrar dados IBGE (se disponíveis)
    ###################################################################################################################
    
    ibge_file = os.path.join(PROCESSED_DATA_DIR, 'ibge_covid_variables.csv')
    if os.path.exists(ibge_file):
        data_logger.info("Integrando dados IBGE...")
        try:
            df_ibge = pd.read_csv(ibge_file)
            data_logger.info(f"Dados IBGE carregados: {df_ibge.shape}")
            
            # Verificar se temos código do município para fazer o merge
            if 'city_ibge_code' in df_final_2020.columns and 'CodMun' in df_ibge.columns:
                # Merge por código IBGE
                df_final_2020 = pd.merge(
                    df_final_2020,
                    df_ibge,
                    left_on='city_ibge_code',
                    right_on='CodMun',
                    how='left'
                )
                df_final_2020 = df_final_2020.drop(columns=['CodMun'], errors='ignore')
                data_logger.info(f"Integração IBGE por código: {df_final_2020.shape}")
            else:
                data_logger.warning("Não foi possível integrar dados IBGE (falta city_ibge_code ou CodMun)")
                
        except Exception as e:
            data_logger.error(f"Erro ao integrar dados IBGE: {e}")
    else:
        data_logger.warning(f"Arquivo IBGE não encontrado: {ibge_file}")
    
    ###################################################################################################################
    # Passo 3.2: Integrar dados PIB (se disponíveis)
    ###################################################################################################################
    
    pib_file = os.path.join(PROCESSED_DATA_DIR, 'pib_municipal_integration.csv')
    if os.path.exists(pib_file):
        data_logger.info("Integrando dados PIB...")
        try:
            df_pib = pd.read_csv(pib_file)
            data_logger.info(f"Dados PIB carregados: {df_pib.shape}")
            
            # Verificar se temos código do município para fazer o merge
            pib_code_col = 'cod_municipio' if 'cod_municipio' in df_pib.columns else 'CodMun'
            
            if 'city_ibge_code' in df_final_2020.columns and pib_code_col in df_pib.columns:
                # Merge por código IBGE
                df_final_2020 = pd.merge(
                    df_final_2020,
                    df_pib,
                    left_on='city_ibge_code',
                    right_on=pib_code_col,
                    how='left'
                )
                df_final_2020 = df_final_2020.drop(columns=[pib_code_col, 'municipio_y', 'uf_y'], errors='ignore')
                # Renomear colunas duplicadas se necessário
                if 'municipio_x' in df_final_2020.columns:
                    df_final_2020 = df_final_2020.rename(columns={'municipio_x': 'municipio'})
                if 'uf_x' in df_final_2020.columns:
                    df_final_2020 = df_final_2020.rename(columns={'uf_x': 'uf'})
                data_logger.info(f"Integração PIB por código: {df_final_2020.shape}")
            else:
                data_logger.warning(f"Não foi possível integrar dados PIB (falta city_ibge_code ou {pib_code_col})")
                
        except Exception as e:
            data_logger.error(f"Erro ao integrar dados PIB: {e}")
    else:
        data_logger.warning(f"Arquivo PIB não encontrado: {pib_file}")
    
    data_logger.info(f"Dataset final após todas as integrações: {df_final_2020.shape}")
    
    # Normalizar nomes de todas as colunas antes de salvar
    normalized_columns = {}
    for col in df_final_2020.columns:
        normalized_col = normalize_variable_name(col)
        if normalized_col != col:
            normalized_columns[col] = normalized_col
    
    if normalized_columns:
        df_final_2020 = df_final_2020.rename(columns=normalized_columns)
        data_logger.info(f"Normalizando {len(normalized_columns)} nomes de colunas do dataset final")
    
    # Aplicar limpeza inteligente de colunas duplicadas (_x, _y)
    df_final_2020 = remove_duplicate_columns(df_final_2020, data_logger)
    
    df_final_2020.shape[0]
    # 5443
    df_final_2020 = df_final_2020.drop(columns=['origem'])

    df_final_2020.columns

    df_final_2020 = df_final_2020.dropna()

# df_final_2020.to_csv('df_final.csv', index=False)
# df_final_2020.to_excel('df_final.xlsx', index=False, sheet_name='Planilha1')
    # Verificar e corrigir tipos de dados antes de salvar Parquet
    data_logger.info("Verificando tipos de dados antes de salvar Parquet...")
    
    # Converter colunas object que ainda restaram para string
    object_cols = df_final_2020.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        if col not in ['uf', 'municipio']:  # Manter algumas colunas como object
            try:
                df_final_2020[col] = df_final_2020[col].astype(str)
                data_logger.info(f"Convertido {col} para string")
            except Exception as e:
                data_logger.warning(f"Erro ao converter {col} para string: {e}")
                # Se não conseguir converter, remover a coluna problemática
                df_final_2020 = df_final_2020.drop(columns=[col])
                data_logger.warning(f"Coluna {col} removida devido a problemas de tipo")
    
    output_path = os.path.join(PROCESSED_DATA_DIR, 'df_final_2020.parquet')
    df_final_2020.to_parquet(output_path, index=False)
    print(f"Dados gerais processados e salvos em: {output_path}")

    # --- Create and save UF-specific files ---
    print("\nIniciando criação de arquivos Parquet específicos por UF...")
    unique_ufs = df_final_2020['uf'].unique()
    categorical_cols_for_uf_specific = [
        'cor_raca', 'estado_civil', 'faixa_etaria', 'genero',
        'grau_de_instrucao', 'ocupacao', 'partido', 'regiao'
    ]

    for uf_value in unique_ufs:
        print(f"  Processando UF: {uf_value}...")
        df_uf_specific = df_final_2020[df_final_2020['uf'] == uf_value].copy()
        
        # Drop the original 'uf' column as it's now implicit to the file/segment
        df_uf_specific.drop(columns=['uf'], inplace=True)
        
        # Dummify other categorical columns for this UF-specific DataFrame
        # Ensure all expected categorical columns exist before trying to dummify
        cols_to_dummify_present = [col for col in categorical_cols_for_uf_specific if col in df_uf_specific.columns]
        if cols_to_dummify_present:
            # Padronizar valores omitidos antes de criar dummies
            for col in cols_to_dummify_present:
                # Primeiro tratar valores NaN reais (antes de converter para string)
                df_uf_specific[col] = df_uf_specific[col].fillna('Omitido')
                df_uf_specific[col] = df_uf_specific[col].astype(str)
                
                # Tratar casos onde NaN virou string "nan" durante astype(str)
                df_uf_specific[col] = df_uf_specific[col].replace('nan', 'Omitido')
                
                # Tratamento específico para campo Mcov01 (isolamento social)
                if 'Mcov01' in col:
                    mcov01_mappings = {
                        'Sim, a população foi orientada para permanecer em isolamento social': 'orientacao_isolamento',
                        'Sim, foi decretado isolamento social': 'decretado_isolamento',
                        'Recusa': 'Omitido',
                        'Não': 'nao_ao_isolamento'
                    }
                    for original_val, new_val in mcov01_mappings.items():
                        # Normalizar o valor mapeado
                        normalized_val = normalize_variable_name(new_val)
                        df_uf_specific[col] = df_uf_specific[col].replace(original_val, normalized_val)
                        if original_val in df_uf_specific[col].values:
                            data_logger.info(f"Mapeando Mcov01 para UF {uf_value}: '{original_val}' -> '{normalized_val}'")
                
                # Tratamento específico para campo Mmam01 (órgão gestor meio ambiente)
                if 'Mmam01' in col:
                    mmam01_mappings = {
                        'Não possui estrutura': 'nao_possui_orgao_gestor',
                        'Órgão da administração indireta': 'administracao_indireta_orgao_gestor',
                        'Secretaria em conjunto com outras políticas setoriais': 'em_conjunto_orgao_gestor',
                        'Secretaria exclusiva': 'exclusivo_orgao_gestor',
                        'Setor subordinado a outra secretaria': 'subordinado_outro_orgao_gestor',
                        'Setor subordinado diretamente à chefia do Executivo': 'subordinado_executivo_orgao_gestor'
                    }
                    for original_val, new_val in mmam01_mappings.items():
                        # Normalizar o valor mapeado
                        normalized_val = normalize_variable_name(new_val)
                        df_uf_specific[col] = df_uf_specific[col].replace(original_val, normalized_val)
                        if original_val in df_uf_specific[col].values:
                            data_logger.info(f"Mapeando Mmam01 para UF {uf_value}: '{original_val}' -> '{normalized_val}'")
                
                # Tratar strings representando valores ausentes (incluindo Recusa para outros campos)
                missing_values = ['-', 'nan', 'None', 'Não informou', 'Não informado', 'NaN', 'null', 'NULL', '', 'Recusa']
                for missing_val in missing_values:
                    df_uf_specific[col] = df_uf_specific[col].replace(missing_val, 'Omitido')
                
                # Normalizar todos os valores categóricos (remover acentos e espaços)
                unique_vals = df_uf_specific[col].unique()
                for val in unique_vals:
                    if isinstance(val, str) and val not in ['Omitido']:  # Não normalizar 'Omitido'
                        normalized_val = normalize_variable_name(val)
                        if normalized_val != val:  # Só substituir se houve mudança
                            df_uf_specific[col] = df_uf_specific[col].replace(val, normalized_val)
                            data_logger.info(f"Normalizando valor em {col} para UF {uf_value}: '{val}' -> '{normalized_val}'")
                
                # Truncar nomes de valores ainda muito longos (>50 caracteres) - após mapeamentos
                unique_vals = df_uf_specific[col].unique()
                for val in unique_vals:
                    if isinstance(val, str) and len(val) > 50:
                        truncated_val = val[:47] + "..."
                        df_uf_specific[col] = df_uf_specific[col].replace(val, truncated_val)
                        data_logger.info(f"Truncando valor longo em {col} para UF {uf_value}: '{val}' -> '{truncated_val}'")
                
                data_logger.info(f"Padronizando valores omitidos em {col} para UF {uf_value}")
            
            df_uf_specific = pd.get_dummies(
                df_uf_specific, 
                columns=cols_to_dummify_present, 
                drop_first=False, # Consistent with current general dummification in models.py
                dummy_na=False,  # Não criar coluna para NaN pois já tratamos como 'Omitido'
                dtype='int'
            )
            
            # Normalizar nomes das colunas dummy criadas para UF-específico
            normalized_columns = {}
            for col in df_uf_specific.columns:
                if any(cat_col in col for cat_col in cols_to_dummify_present):  # Se é coluna dummy criada
                    normalized_col = normalize_variable_name(col)
                    if normalized_col != col:
                        normalized_columns[col] = normalized_col
            
            if normalized_columns:
                df_uf_specific = df_uf_specific.rename(columns=normalized_columns)
                data_logger.info(f"Normalizando {len(normalized_columns)} nomes de colunas dummy para UF {uf_value}")
        else:
            print(f"    Nenhuma coluna categórica (além de 'uf') para dummificar para {uf_value}.")

        output_path_uf = os.path.join(PROCESSED_DATA_DIR, f'df_final_2020_{uf_value}.parquet')
        df_uf_specific.to_parquet(output_path_uf, index=False)
        print(f"    Dados processados para UF {uf_value} salvos em: {output_path_uf}")
    
    print("Criação de arquivos Parquet específicos por UF concluída.")

if __name__ == '__main__':
    process_and_save_data()


