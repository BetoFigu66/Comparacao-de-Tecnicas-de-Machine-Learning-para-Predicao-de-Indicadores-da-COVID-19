# -*- coding: utf-8 -*-
"""
Script para analisar quantas features resultaram dos dados dos perfis dos municípios brasileiros
"""

import pandas as pd
import os
from collections import defaultdict

def analyze_features():
    """
    Analisa as features dos dados dos perfis dos municípios
    """
    
    # Caminho para o arquivo principal de dados
    data_path = r"c:\Beto\Profissional\MBA_UspEsalq\TCC\git\Prj_TCC_JRSF_MBA_USP_ESALQ\Data\Processed\df_final_2020.parquet"
    
    if not os.path.exists(data_path):
        print(f"ERRO: Arquivo nao encontrado: {data_path}")
        return
    
    print("Carregando dados dos perfis dos municipios...")
    df = pd.read_parquet(data_path)
    
    print(f"Total de colunas no dataset: {len(df.columns)}")
    print(f"Total de municipios: {len(df)}")
    
    # Categorizar as features por prefixo
    feature_categories = defaultdict(list)
    
    for col in df.columns:
        if col.startswith('IBGE_'):
            feature_categories['IBGE'].append(col)
        elif col.startswith('ECONOMICOS_'):
            feature_categories['ECONOMICOS'].append(col)
        elif col.startswith('ELEITORAIS_'):
            feature_categories['ELEITORAIS'].append(col)
        else:
            # Outras categorias (identificadores, target, etc.)
            if col in ['municipio', 'city_ibge_code', 'uf', 'regiao', 'codigo_ibge']:
                feature_categories['IDENTIFICADORES'].append(col)
            elif col in ['target_eleicao_2020', 'target_eleicao_2024']:
                feature_categories['TARGET'].append(col)
            else:
                feature_categories['OUTRAS'].append(col)
    
    print("\n" + "="*80)
    print("ANALISE DETALHADA DAS FEATURES POR CATEGORIA")
    print("="*80)
    
    total_features = 0
    
    for category, features in feature_categories.items():
        count = len(features)
        total_features += count
        print(f"\n{category}: {count} features")
        
        if category == 'IBGE':
            # Subcategorizar IBGE por tipo
            ibge_subcategories = defaultdict(list)
            for feature in features:
                if 'COVID_19' in feature:
                    ibge_subcategories['COVID-19'].append(feature)
                elif 'Meio_ambiente' in feature:
                    ibge_subcategories['Meio Ambiente'].append(feature)
                elif 'Recursos_humanos' in feature:
                    ibge_subcategories['Recursos Humanos'].append(feature)
                elif 'Financas' in feature:
                    ibge_subcategories['Finanças'].append(feature)
                elif 'Educacao' in feature:
                    ibge_subcategories['Educação'].append(feature)
                elif 'Saude' in feature:
                    ibge_subcategories['Saúde'].append(feature)
                elif 'Assistencia_social' in feature:
                    ibge_subcategories['Assistência Social'].append(feature)
                elif 'Cultura' in feature:
                    ibge_subcategories['Cultura'].append(feature)
                elif 'Esporte' in feature:
                    ibge_subcategories['Esporte'].append(feature)
                elif 'Habitacao' in feature:
                    ibge_subcategories['Habitação'].append(feature)
                elif 'Saneamento' in feature:
                    ibge_subcategories['Saneamento'].append(feature)
                elif 'Transporte' in feature:
                    ibge_subcategories['Transporte'].append(feature)
                else:
                    ibge_subcategories['Outros IBGE'].append(feature)
            
            for subcat, subfeatures in ibge_subcategories.items():
                print(f"   -> {subcat}: {len(subfeatures)} features")
        
        elif category == 'ELEITORAIS':
            # Subcategorizar eleitorais
            eleitorais_subcategories = defaultdict(list)
            for feature in features:
                if 'partido_' in feature:
                    eleitorais_subcategories['Partidos'].append(feature)
                elif 'cor_raca_' in feature:
                    eleitorais_subcategories['Cor/Raça'].append(feature)
                elif 'genero_' in feature:
                    eleitorais_subcategories['Gênero'].append(feature)
                elif 'grau_instrucao_' in feature:
                    eleitorais_subcategories['Grau de Instrução'].append(feature)
                elif 'estado_civil_' in feature:
                    eleitorais_subcategories['Estado Civil'].append(feature)
                elif 'ocupacao_' in feature:
                    eleitorais_subcategories['Ocupação'].append(feature)
                elif feature.startswith('uf_') or feature.startswith('regiao_'):
                    eleitorais_subcategories['Geografia'].append(feature)
                else:
                    eleitorais_subcategories['Outros Eleitorais'].append(feature)
            
            for subcat, subfeatures in eleitorais_subcategories.items():
                print(f"   -> {subcat}: {len(subfeatures)} features")
    
    print(f"\nTOTAL GERAL: {total_features} features")
    
    # Análise específica das features dos perfis dos municípios (IBGE)
    ibge_features = feature_categories['IBGE']
    print(f"\nFOCO: Features dos Perfis dos Municipios (IBGE): {len(ibge_features)} features")
    
    # Salvar relatório detalhado
    output_file = r"c:\Beto\Profissional\MBA_UspEsalq\TCC\git\Prj_TCC_JRSF_MBA_USP_ESALQ\analise_features_perfis_municipios.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ANÁLISE DETALHADA DAS FEATURES DOS PERFIS DOS MUNICÍPIOS BRASILEIROS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data da análise: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arquivo analisado: {data_path}\n")
        f.write(f"Total de municípios: {len(df):,}\n")
        f.write(f"Total de features: {len(df.columns):,}\n\n")
        
        f.write("RESUMO POR CATEGORIA:\n")
        f.write("-" * 40 + "\n")
        
        for category, features in feature_categories.items():
            f.write(f"{category}: {len(features)} features\n")
        
        f.write(f"\nTOTAL: {total_features} features\n\n")
        
        f.write("DETALHAMENTO DAS FEATURES IBGE (PERFIS DOS MUNICÍPIOS):\n")
        f.write("-" * 60 + "\n")
        
        # Subcategorizar IBGE detalhadamente
        ibge_subcategories = defaultdict(list)
        for feature in ibge_features:
            if 'COVID_19' in feature:
                ibge_subcategories['COVID-19'].append(feature)
            elif 'Meio_ambiente' in feature:
                ibge_subcategories['Meio Ambiente'].append(feature)
            elif 'Recursos_humanos' in feature:
                ibge_subcategories['Recursos Humanos'].append(feature)
            elif 'Financas' in feature:
                ibge_subcategories['Finanças'].append(feature)
            elif 'Educacao' in feature:
                ibge_subcategories['Educação'].append(feature)
            elif 'Saude' in feature:
                ibge_subcategories['Saúde'].append(feature)
            elif 'Assistencia_social' in feature:
                ibge_subcategories['Assistência Social'].append(feature)
            elif 'Cultura' in feature:
                ibge_subcategories['Cultura'].append(feature)
            elif 'Esporte' in feature:
                ibge_subcategories['Esporte'].append(feature)
            elif 'Habitacao' in feature:
                ibge_subcategories['Habitação'].append(feature)
            elif 'Saneamento' in feature:
                ibge_subcategories['Saneamento'].append(feature)
            elif 'Transporte' in feature:
                ibge_subcategories['Transporte'].append(feature)
            else:
                ibge_subcategories['Outros IBGE'].append(feature)
        
        for subcat, subfeatures in sorted(ibge_subcategories.items()):
            f.write(f"\n{subcat}: {len(subfeatures)} features\n")
            for i, feature in enumerate(sorted(subfeatures), 1):
                f.write(f"  {i:3d}. {feature}\n")
        
        f.write(f"\nTOTAL FEATURES IBGE: {len(ibge_features)}\n")
    
    print(f"\nRelatorio detalhado salvo em: {output_file}")
    
    return {
        'total_features': total_features,
        'ibge_features': len(ibge_features),
        'economicos_features': len(feature_categories['ECONOMICOS']),
        'eleitorais_features': len(feature_categories['ELEITORAIS']),
        'categories': dict(feature_categories)
    }

if __name__ == "__main__":
    result = analyze_features()
    
    print("\n" + "="*80)
    print("RESPOSTA A PERGUNTA:")
    print("="*80)
    print(f"Features dos Perfis dos Municipios Brasileiros (IBGE): {result['ibge_features']} features")
    print(f"Features Economicas: {result['economicos_features']} features")
    print(f"Features Eleitorais: {result['eleitorais_features']} features")
    print(f"Total Geral de Features: {result['total_features']} features")
