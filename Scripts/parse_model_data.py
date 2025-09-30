"""
Script para análise e processamento de dados dos modelos treinados
Funcionalidades:
1. Atualizar métricas dos modelos (MSE, RMSE, MAE, MAPE)
2. Gerar CSV com feature importance por modelo
"""

# Suprimir logs do TensorFlow e warnings
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir INFO, WARNING e ERROR do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilitar oneDNN
warnings.filterwarnings('ignore')  # Suprimir warnings do scikit-learn

import json
import joblib
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime

# Adicionar o diretório Models ao path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
models_dir = os.path.join(project_root, 'Models')
sys.path.append(models_dir)

from models import load_and_prepare_data

def calculate_mape(y_true, y_pred):
    """Calcula Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Evitar divisão por zero
    mask = y_true != 0
    if not mask.any():
        return float('inf')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_additional_metrics(model, X_test, y_test):
    """Calcula MSE, RMSE, MAE e MAPE para um modelo"""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }

def update_model_metrics():
    """Atualiza as métricas dos modelos já treinados"""
    
    print("=== ATUALIZAR MÉTRICAS DE MODELOS ===")
    print("Adicionando MSE, RMSE, MAE e MAPE aos modelos já treinados...")
    print()
    
    # Caminhos
    trained_models_dir = os.path.join(project_root, 'Models', 'trainedModels')
    log_file = os.path.join(trained_models_dir, 'trained_models_log.json')
    
    if not os.path.exists(log_file):
        print(f"[ERRO] Arquivo {log_file} não encontrado!")
        return
    
    # Carregar log existente
    with open(log_file, 'r') as f:
        model_logs = json.load(f)
    
    print(f"Encontrados {len(model_logs)} modelos para atualizar...")
    
    updated_count = 0
    
    for i, model_entry in enumerate(model_logs):
        try:
            print(f"\n{i+1}/{len(model_logs)} - Processando {model_entry['model_name']}...")
            
            # Verificar se já tem as métricas
            if all(key in model_entry for key in ['mse', 'rmse', 'mae', 'mape']):
                print("  [OK] Métricas já existem, pulando...")
                continue
            
            # Carregar modelo
            model_path = model_entry['saved_model_path']
            
            # Converter caminho Linux para Windows se necessário (apenas no Windows)
            if os.name == 'nt' and model_path.startswith('/mnt/c/'):  # 'nt' = Windows
                model_path = model_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
            
            if not os.path.exists(model_path):
                print(f"  [ERRO] Arquivo do modelo não encontrado: {model_path}")
                continue
            
            model = joblib.load(model_path)
            print(f"  [LOAD] Modelo carregado: {os.path.basename(model_path)}")
            
            # Determinar contexto dos dados
            data_context = model_entry.get('data_context', 'Geral (todas UFs)')
            uf_target = None
            
            if data_context.startswith("UF: "):
                uf_target = data_context.split(": ")[1]
            
            # Carregar dados correspondentes
            print(f"  [DATA] Carregando dados: {data_context}")
            X, y_cases, y_deaths, feature_names, scaler, _ = load_and_prepare_data(
                uf_target=uf_target, 
                include_ibge=True, 
                include_pib=True
            )
            
            # Determinar target variable
            target_var = model_entry.get('target_variable', 'cases')
            y = y_cases if target_var == 'cases' else y_deaths
            
            # Split dos dados (mesmo random_state usado no treinamento)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Calcular métricas adicionais
            print("  [CALC] Calculando métricas...")
            additional_metrics = calculate_additional_metrics(model, X_test, y_test)
            
            # Atualizar entrada do log
            model_entry.update(additional_metrics)
            updated_count += 1
            
            print(f"  [DONE] Métricas atualizadas:")
            print(f"    MSE: {additional_metrics['mse']:.4f}")
            print(f"    RMSE: {additional_metrics['rmse']:.4f}")
            print(f"    MAE: {additional_metrics['mae']:.4f}")
            print(f"    MAPE: {additional_metrics['mape']:.2f}%")
            
        except Exception as e:
            print(f"  [ERRO] Erro ao processar {model_entry.get('model_name', 'modelo')}: {e}")
            continue
    
    # Salvar log atualizado
    if updated_count > 0:
        # Backup do arquivo original
        backup_file = log_file.replace('.json', '_backup.json')
        with open(backup_file, 'w') as f:
            json.dump(model_logs, f, indent=4)
        print(f"\n[BACKUP] Backup salvo em: {backup_file}")
        
        # Salvar arquivo atualizado
        with open(log_file, 'w') as f:
            json.dump(model_logs, f, indent=4)
        
        print(f"[SUCCESS] {updated_count} modelos atualizados com sucesso!")
        print(f"[SAVE] Log atualizado salvo em: {log_file}")
    else:
        print("\n[INFO] Nenhum modelo precisou ser atualizado.")

def generate_feature_importance_csv():
    """Gera CSV com feature importance de todos os modelos"""
    
    print("=== GERAR CSV FEATURE IMPORTANCE ===")
    print("Criando matriz: Features x Modelos com importâncias...")
    print()
    
    # Caminhos
    trained_models_dir = os.path.join(project_root, 'Models', 'trainedModels')
    log_file = os.path.join(trained_models_dir, 'trained_models_log.json')
    
    if not os.path.exists(log_file):
        print(f"[ERRO] Arquivo {log_file} não encontrado!")
        return
    
    # Carregar log existente
    with open(log_file, 'r') as f:
        model_logs = json.load(f)
    
    print(f"Processando {len(model_logs)} modelos...")
    
    # Dicionário para armazenar importâncias: {feature_name: {model_id: importance}}
    feature_importance_matrix = {}
    model_columns = []
    
    for i, model_entry in enumerate(model_logs):
        try:
            model_name = model_entry['model_name']
            timestamp = model_entry['timestamp']
            data_context = model_entry.get('data_context', 'Geral')
            target_var = model_entry.get('target_variable', 'cases')
            
            # Criar identificador único do modelo
            model_id = f"{model_name}_{target_var}_{data_context}_{timestamp}"
            model_columns.append(model_id)
            
            print(f"  [{i+1}/{len(model_logs)}] Processando {model_name}...")
            
            # Extrair feature importances
            feature_importances = model_entry.get('feature_importances', [])
            
            if not feature_importances:
                print(f"    [AVISO] Sem feature importances para {model_name}")
                continue
            
            # Processar cada feature importance
            for feat_info in feature_importances:
                if isinstance(feat_info, dict) and 'feature' in feat_info and 'importance' in feat_info:
                    feature_name = feat_info['feature']
                    importance = feat_info['importance']
                    
                    # Inicializar feature se não existir
                    if feature_name not in feature_importance_matrix:
                        feature_importance_matrix[feature_name] = {}
                    
                    # Armazenar importância
                    feature_importance_matrix[feature_name][model_id] = importance
            
            print(f"    [OK] {len(feature_importances)} features processadas")
            
        except Exception as e:
            print(f"    [ERRO] Erro ao processar modelo {i+1}: {e}")
            continue
    
    if not feature_importance_matrix:
        print("[ERRO] Nenhuma feature importance encontrada!")
        return
    
    # Criar DataFrame
    print("\n[BUILD] Construindo DataFrame...")
    
    # Obter todas as features únicas
    all_features = sorted(feature_importance_matrix.keys())
    
    # Criar matriz
    data_matrix = []
    for feature in all_features:
        row = [feature]  # Primeira coluna é o nome da feature
        for model_id in model_columns:
            importance = feature_importance_matrix[feature].get(model_id, 0.0)
            row.append(importance)
        data_matrix.append(row)
    
    # Criar DataFrame
    columns = ['Feature'] + model_columns
    df = pd.DataFrame(data_matrix, columns=columns)
    
    # Salvar CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'feature_importance_matrix_{timestamp}.csv'
    csv_path = os.path.join(trained_models_dir, csv_filename)
    
    df.to_csv(csv_path, index=False)
    
    print(f"[SUCCESS] CSV gerado com sucesso!")
    print(f"[SAVE] Arquivo salvo em: {csv_path}")
    print(f"[INFO] Dimensões: {len(all_features)} features x {len(model_columns)} modelos")
    print(f"[INFO] Total de células: {len(all_features) * len(model_columns)}")
    
    # Mostrar estatísticas
    print(f"\n[STATS] Estatísticas:")
    print(f"  - Features com dados: {len(all_features)}")
    print(f"  - Modelos processados: {len(model_columns)}")
    print(f"  - Células não-zero: {(df.iloc[:, 1:] != 0).sum().sum()}")

def list_predictor_variables():
    """Lista todas as variáveis preditoras do dataset"""
    
    print("=== LISTAR VARIÁVEIS PREDITORAS ===")
    print("Analisando variáveis disponíveis no dataset...")
    print()
    
    # Caminho do arquivo parquet
    data_path = os.path.join(project_root, 'Data', 'Processed', 'df_final_2020.parquet')
    
    if not os.path.exists(data_path):
        print(f"[ERRO] Arquivo não encontrado: {data_path}")
        return
    
    try:
        # Carregar dados
        print("[LOAD] Carregando dados do parquet...")
        df = pd.read_parquet(data_path)
        
        print(f"[INFO] Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
        
        # Separar variáveis por categoria
        all_columns = df.columns.tolist()
        
        # Remover variáveis target e identificadores
        exclude_vars = ['cases_per_100k', 'deaths_per_100k', 'city_ibge_code', 'municipio', 'uf']
        predictor_vars = [col for col in all_columns if col not in exclude_vars]
        
        print(f"[INFO] Total de variáveis preditoras: {len(predictor_vars)}")
        print()
        
        # Categorizar variáveis
        categories = {
            'IBGE': [],
            'PIB': [],
            'Eleitorais': [],
            'COVID/População': [],
            'Geográficas': [],
            'Outras': []
        }
        
        for var in predictor_vars:
            var_lower = var.lower()
            if var.startswith('IBGE_'):
                categories['IBGE'].append(var)
            elif var.startswith('PIB_'):
                categories['PIB'].append(var)
            elif any(term in var_lower for term in ['partido', 'cor_raca', 'genero', 'ocupacao', 'estado_civil']):
                categories['Eleitorais'].append(var)
            elif any(term in var_lower for term in ['populacao', 'densidade', 'covid']):
                categories['COVID/População'].append(var)
            elif any(term in var_lower for term in ['latitude', 'longitude', 'regiao', 'capital']):
                categories['Geográficas'].append(var)
            else:
                categories['Outras'].append(var)
        
        # Exibir por categoria
        for category, vars_list in categories.items():
            if vars_list:
                print(f"\n{'='*60}")
                print(f"📊 {category.upper()} ({len(vars_list)} variáveis)")
                print('='*60)
                
                for i, var in enumerate(sorted(vars_list), 1):
                    # Mostrar tipo de dados e valores únicos para variáveis categóricas
                    dtype = df[var].dtype
                    n_unique = df[var].nunique()
                    
                    if dtype == 'object' or n_unique <= 10:
                        unique_vals = df[var].unique()[:5]  # Primeiros 5 valores únicos
                        unique_str = ', '.join([str(v) for v in unique_vals])
                        if len(unique_vals) < n_unique:
                            unique_str += f", ... (+{n_unique - len(unique_vals)} mais)"
                        print(f"  {i:3d}. {var:<50} | {dtype} | {n_unique} únicos | Ex: {unique_str}")
                    else:
                        min_val = df[var].min()
                        max_val = df[var].max()
                        print(f"  {i:3d}. {var:<50} | {dtype} | Range: {min_val:.2f} - {max_val:.2f}")
        
        # Salvar lista em arquivo texto
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(project_root, 'Models', 'trainedModels', f'predictor_variables_{timestamp}.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("LISTA DE VARIÁVEIS PREDITORAS\n")
            f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {data_path}\n")
            f.write(f"Total de variáveis: {len(predictor_vars)}\n")
            f.write("="*80 + "\n\n")
            
            for category, vars_list in categories.items():
                if vars_list:
                    f.write(f"\n{category.upper()} ({len(vars_list)} variáveis)\n")
                    f.write("-" * 60 + "\n")
                    for var in sorted(vars_list):
                        dtype = df[var].dtype
                        n_unique = df[var].nunique()
                        f.write(f"{var} | {dtype} | {n_unique} valores únicos\n")
        
        print(f"\n[SAVE] Lista salva em: {output_file}")
        
        # Estatísticas gerais
        print(f"\n{'='*60}")
        print("📈 ESTATÍSTICAS GERAIS")
        print('='*60)
        for category, vars_list in categories.items():
            if vars_list:
                print(f"  {category:<15}: {len(vars_list):4d} variáveis")
        print(f"  {'TOTAL':<15}: {len(predictor_vars):4d} variáveis")
        
        # Identificar possíveis problemas
        print(f"\n{'='*60}")
        print("⚠️  POSSÍVEIS PROBLEMAS IDENTIFICADOS")
        print('='*60)
        
        problematic_vars = []
        # Variáveis conhecidas que devem ser ignoradas na análise de problemas
        ignore_high_unique = ['estimated_population', 'last_available_confirmed_per_100k_inhabitants', 'death_per_100k_inhabitants']
        
        for var in predictor_vars:
            # Variáveis com nomes muito longos
            if len(var) > 80:
                problematic_vars.append(f"Nome muito longo: {var}")
            
            # Variáveis com sufixos estranhos (_x, _y)
            if var.endswith('_x') or var.endswith('_y'):
                problematic_vars.append(f"Sufixo suspeito: {var}")
            
            # Variáveis com muitos valores únicos (possível vazamento) - ignorar as conhecidas
            if df[var].nunique() > len(df) * 0.8 and var not in ignore_high_unique:
                problematic_vars.append(f"Muitos valores únicos: {var} ({df[var].nunique()} únicos)")
        
        if problematic_vars:
            for i, problem in enumerate(problematic_vars[:20], 1):  # Mostrar apenas os primeiros 20
                print(f"  {i:2d}. {problem}")
            if len(problematic_vars) > 20:
                print(f"  ... e mais {len(problematic_vars) - 20} problemas")
        else:
            print("  Nenhum problema óbvio identificado.")
        
    except Exception as e:
        print(f"[ERRO] Erro ao processar dados: {e}")

def show_menu():
    """Exibe menu principal"""
    print("\n" + "="*50)
    print("    ANÁLISE DE DADOS DOS MODELOS TREINADOS")
    print("="*50)
    print("1. Atualizar métricas de modelos (MSE, RMSE, MAE, MAPE)")
    print("2. Gerar CSV com Feature Importance por modelo")
    print("3. Listar todas as variáveis preditoras")
    print("0. Sair")
    print("-"*50)

def main():
    """Função principal com menu"""
    
    while True:
        show_menu()
        
        try:
            choice = input("Escolha uma opção: ").strip()
            
            if choice == '0':
                print("\n[EXIT] Saindo...")
                break
            elif choice == '1':
                print()
                update_model_metrics()
                input("\nPressione Enter para continuar...")
            elif choice == '2':
                print()
                generate_feature_importance_csv()
                input("\nPressione Enter para continuar...")
            elif choice == '3':
                print()
                list_predictor_variables()
                input("\nPressione Enter para continuar...")
            else:
                print("\n[ERRO] Opção inválida! Escolha 0, 1, 2 ou 3.")
                
        except KeyboardInterrupt:
            print("\n\n[EXIT] Interrompido pelo usuário. Saindo...")
            break
        except Exception as e:
            print(f"\n[ERRO] Erro inesperado: {e}")
            input("Pressione Enter para continuar...")

if __name__ == "__main__":
    main()
