"""
Script para processar dados do IBGE e criar arquivo ibge_covid_variables.csv
para uso nos modelos de predição de COVID-19

Autor: TCC Analysis
Data: 2025-09-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ibge_data():
    """Processa dados do IBGE e cria arquivo ibge_covid_variables.csv"""
    
    # Definir caminhos
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    ibge_file = project_root / "DataSources" / "IBGE" / "Base_MUNIC_2020.xlsx"
    output_file = project_root / "Data" / "Processed" / "ibge_covid_variables.csv"
    
    # Verificar se arquivo IBGE existe
    if not ibge_file.exists():
        logger.error(f"Arquivo IBGE não encontrado: {ibge_file}")
        return False
    
    logger.info(f"Processando dados IBGE de: {ibge_file}")
    
    try:
        # Carregar dados do IBGE
        xl = pd.ExcelFile(ibge_file)
        logger.info(f"Sheets disponíveis: {xl.sheet_names}")
        
        # Variáveis selecionadas baseadas na análise de relevância para COVID-19
        selected_variables = {
            # Infraestrutura de saúde
            'A12': ['CodMun', 'A12001', 'A12002', 'A12003', 'A12004'],  # Estabelecimentos de saúde
            'A13': ['CodMun', 'A13001', 'A13002', 'A13003', 'A13004'],  # Profissionais de saúde
            
            # Demografia e vulnerabilidade social
            'A1': ['CodMun', 'A1001', 'A1002', 'A1003'],  # População
            'A2': ['CodMun', 'A2001', 'A2002', 'A2003'],  # Densidade populacional
            
            # Programas sociais
            'A14': ['CodMun', 'A14001', 'A14002', 'A14003'],  # Programas de assistência social
            'A15': ['CodMun', 'A15001', 'A15002'],  # Bolsa Família
            
            # Educação
            'A11': ['CodMun', 'A11001', 'A11002', 'A11003'],  # Estabelecimentos de ensino
            
            # Segurança alimentar
            'A16': ['CodMun', 'A16001', 'A16002'],  # Programas de segurança alimentar
            
            # Economia
            'A17': ['CodMun', 'A17001', 'A17002', 'A17003'],  # Indicadores econômicos
        }
        
        combined_data = None
        
        # Processar cada sheet
        for sheet_name, columns in selected_variables.items():
            try:
                if sheet_name not in xl.sheet_names:
                    logger.warning(f"Sheet {sheet_name} não encontrada")
                    continue
                
                logger.info(f"Processando sheet: {sheet_name}")
                df = pd.read_excel(ibge_file, sheet_name=sheet_name)
                
                # Verificar se as colunas existem
                available_cols = ['CodMun']  # Sempre incluir código do município
                for col in columns[1:]:  # Pular CodMun
                    if col in df.columns:
                        available_cols.append(col)
                    else:
                        logger.warning(f"Coluna {col} não encontrada em {sheet_name}")
                
                if len(available_cols) > 1:  # Se temos mais que só CodMun
                    sheet_data = df[available_cols].copy()
                    
                    # Renomear colunas para incluir prefixo do sheet
                    rename_dict = {}
                    for col in available_cols:
                        if col != 'CodMun':
                            rename_dict[col] = f'IBGE_{sheet_name}_{col}'
                    
                    sheet_data = sheet_data.rename(columns=rename_dict)
                    
                    # Combinar com dados existentes
                    if combined_data is None:
                        combined_data = sheet_data
                    else:
                        combined_data = combined_data.merge(sheet_data, on='CodMun', how='outer')
                    
                    logger.info(f"Adicionadas {len(available_cols)-1} variáveis de {sheet_name}")
                
            except Exception as e:
                logger.error(f"Erro ao processar sheet {sheet_name}: {e}")
                continue
        
        if combined_data is None or combined_data.empty:
            logger.error("Nenhum dado foi processado com sucesso")
            return False
        
        # Limpar dados
        logger.info("Limpando dados...")
        
        # Remover linhas onde CodMun é nulo
        combined_data = combined_data.dropna(subset=['CodMun'])
        
        # Converter CodMun para int
        combined_data['CodMun'] = combined_data['CodMun'].astype(int)
        
        # Tratar valores missing - substituir por 0 onde faz sentido
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'CodMun':
                combined_data[col] = combined_data[col].fillna(0)
        
        # Garantir que diretório de saída existe
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar dados processados
        combined_data.to_csv(output_file, index=False)
        logger.info(f"Dados IBGE salvos em: {output_file}")
        logger.info(f"Shape final: {combined_data.shape}")
        logger.info(f"Variáveis criadas: {len([col for col in combined_data.columns if col.startswith('IBGE_')])}")
        
        # Mostrar resumo das variáveis
        logger.info("Variáveis IBGE disponíveis:")
        for col in sorted(combined_data.columns):
            if col.startswith('IBGE_'):
                non_zero = (combined_data[col] != 0).sum()
                logger.info(f"  {col}: {non_zero} municípios com dados não-zero")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro geral no processamento: {e}")
        return False

if __name__ == "__main__":
    success = process_ibge_data()
    if success:
        print("✅ Processamento dos dados IBGE concluído com sucesso!")
        print("Agora você pode usar include_ibge=True nos modelos.")
    else:
        print("❌ Erro no processamento dos dados IBGE.")
