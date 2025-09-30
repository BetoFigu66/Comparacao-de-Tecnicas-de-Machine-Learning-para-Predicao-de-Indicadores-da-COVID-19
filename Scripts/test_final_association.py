"""
Teste final de associação com arquivo de sinônimos para 100% de sucesso
"""

import pandas as pd
import os

def normalize_municipality_name(name):
    """Normaliza nomes removendo acentos, apóstrofes e underscores"""
    if pd.isna(name):
        return ''
    name = str(name).upper().strip()
    
    # Remover acentos
    replacements = {
        'Á': 'A', 'À': 'A', 'Ã': 'A', 'Â': 'A',
        'É': 'E', 'Ê': 'E',
        'Í': 'I', 'Î': 'I',
        'Ó': 'O', 'Ô': 'O', 'Õ': 'O',
        'Ú': 'U', 'Û': 'U',
        'Ç': 'C'
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remover apóstrofes e underscores
    name = name.replace("'", "").replace("_", "")
    
    return name

def load_synonyms(project_root):
    """Carrega arquivo de sinônimos"""
    synonyms_path = os.path.join(project_root, 'Data', 'municipality_synonyms.csv')
    
    if os.path.exists(synonyms_path):
        synonyms_df = pd.read_csv(synonyms_path)
        # Criar dicionário de mapeamento COVID -> IBGE
        synonym_map = {}
        for _, row in synonyms_df.iterrows():
            covid_key = normalize_municipality_name(row['covid_name']) + row['covid_uf']
            ibge_key = normalize_municipality_name(row['ibge_name']) + row['ibge_uf']
            synonym_map[covid_key] = ibge_key
        
        print(f"Carregados {len(synonym_map)} sinônimos:")
        for covid_key, ibge_key in synonym_map.items():
            print(f"  {covid_key} -> {ibge_key}")
        
        return synonym_map
    else:
        print("Arquivo de sinônimos não encontrado")
        return {}

def test_final_association():
    """Testa associação final com sinônimos"""
    
    # Carregar dados
    project_root = r'C:\Beto\Profissional\MBA_UspEsalq\TCC\Projeto_TCC_Jun_25'
    covid_path = os.path.join(project_root, 'Data', 'Processed', 'df_final_2020.parquet')
    ibge_excel_path = os.path.join(project_root, 'DataSources', 'IBGE', 'Base_MUNIC_2020.xlsx')
    
    covid_df = pd.read_parquet(covid_path)
    ibge_raw = pd.read_excel(ibge_excel_path, sheet_name='Recursos humanos')
    
    # Carregar sinônimos
    synonym_map = load_synonyms(project_root)
    
    # Criar chaves normalizadas
    covid_df['chave_covid'] = covid_df['municipio'].apply(normalize_municipality_name) + covid_df['uf']
    ibge_raw['chave_ibge'] = ibge_raw['Mun'].apply(normalize_municipality_name) + ibge_raw['Sigla UF']
    
    # Aplicar sinônimos nas chaves COVID
    covid_df['chave_covid_final'] = covid_df['chave_covid'].map(lambda x: synonym_map.get(x, x))
    
    chaves_covid_unicas = set(covid_df['chave_covid_final'].unique())
    chaves_ibge_unicas = set(ibge_raw['chave_ibge'].unique())
    
    matches = chaves_covid_unicas.intersection(chaves_ibge_unicas)
    covid_sem_match = sorted(list(chaves_covid_unicas - chaves_ibge_unicas))
    ibge_sem_match = sorted(list(chaves_ibge_unicas - chaves_covid_unicas))
    
    print(f'\n=== ASSOCIACAO FINAL COM SINONIMOS ===')
    print(f'Chaves unicas COVID: {len(chaves_covid_unicas)}')
    print(f'Chaves unicas IBGE: {len(chaves_ibge_unicas)}')
    print(f'Matches encontrados: {len(matches)}')
    print(f'COVID sem match: {len(covid_sem_match)}')
    print(f'IBGE sem match: {len(ibge_sem_match)}')
    
    taxa_sucesso = len(matches) / len(chaves_covid_unicas) * 100
    print(f'Taxa de sucesso: {taxa_sucesso:.2f}%')
    
    if covid_sem_match:
        print(f'\nCOVID sem match no IBGE ({len(covid_sem_match)} municipios):')
        for i, chave in enumerate(covid_sem_match, 1):
            print(f'{i:2d}. {chave}')
    else:
        print('\nSUCESSO TOTAL! Todos os municipios COVID foram associados!')
    
    if ibge_sem_match:
        print(f'\nIBGE sem match no COVID ({len(ibge_sem_match)} municipios):')
        for i, chave in enumerate(ibge_sem_match[:10], 1):  # Mostrar apenas os primeiros 10
            print(f'{i:2d}. {chave}')
        if len(ibge_sem_match) > 10:
            print(f'... e mais {len(ibge_sem_match) - 10} municípios')
    
    return {
        'matches': len(matches),
        'covid_total': len(chaves_covid_unicas),
        'ibge_total': len(chaves_ibge_unicas),
        'taxa_sucesso': taxa_sucesso,
        'covid_sem_match': covid_sem_match,
        'ibge_sem_match': ibge_sem_match,
        'synonym_map': synonym_map
    }

if __name__ == "__main__":
    result = test_final_association()
