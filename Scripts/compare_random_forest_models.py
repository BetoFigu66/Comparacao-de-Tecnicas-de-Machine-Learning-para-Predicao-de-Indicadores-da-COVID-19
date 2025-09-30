"""
Comparação de modelos Random Forest antes e depois da adição das variáveis IBGE
Inclui todas as métricas de avaliação disponíveis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    explained_variance_score,
    max_error,
    mean_squared_log_error
)
import sys
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Add Models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Models'))
from models import load_and_prepare_data

def calculate_all_metrics(y_true, y_pred, model_name="Model"):
    """Calcula todas as métricas de avaliação disponíveis"""
    
    metrics = {}
    
    # Métricas básicas
    metrics['R2'] = r2_score(y_true, y_pred)
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['Explained_Variance'] = explained_variance_score(y_true, y_pred)
    metrics['Max_Error'] = max_error(y_true, y_pred)
    
    # MSLE apenas se todos os valores são positivos
    if np.all(y_true >= 0) and np.all(y_pred >= 0):
        try:
            metrics['MSLE'] = mean_squared_log_error(y_true, y_pred)
        except:
            metrics['MSLE'] = np.nan
    else:
        metrics['MSLE'] = np.nan
    
    # Métricas adicionais
    metrics['Mean_Residual'] = np.mean(y_true - y_pred)
    metrics['Std_Residual'] = np.std(y_true - y_pred)
    
    # Métricas percentuais
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    metrics['MAPE'] = mape
    
    return metrics

def train_and_evaluate_model(X, y, model_name, target_name, include_feature_importance=True):
    """Treina modelo Random Forest e calcula métricas"""
    
    print(f"\n=== {model_name} - {target_name} ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    start_time = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_all_metrics(y_train, y_pred_train, f"{model_name}_train")
    test_metrics = calculate_all_metrics(y_test, y_pred_test, f"{model_name}_test")
    
    # Feature importance
    feature_importance = None
    if include_feature_importance:
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    results = {
        'model_name': model_name,
        'target_name': target_name,
        'training_time': training_time,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance,
        'model': rf_model
    }
    
    # Print results
    print(f"Training time: {training_time:.2f} seconds")
    print(f"\nTrain Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    if feature_importance is not None:
        print(f"\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    return results

def compare_models():
    """Compara modelos Random Forest com e sem variáveis IBGE"""
    
    print("=== COMPARACAO RANDOM FOREST: SEM vs COM VARIAVEIS IBGE ===")
    
    # Carregar dados SEM IBGE
    print("\n1. Carregando dados SEM variáveis IBGE...")
    X_no_ibge, y_cases_no_ibge, y_deaths_no_ibge, features_no_ibge, scaler_no_ibge, _ = load_and_prepare_data(
        uf_target=None, 
        include_ibge=False
    )
    
    # Carregar dados COM IBGE
    print("\n2. Carregando dados COM variáveis IBGE...")
    X_with_ibge, y_cases_with_ibge, y_deaths_with_ibge, features_with_ibge, scaler_with_ibge, _ = load_and_prepare_data(
        uf_target=None, 
        include_ibge=True
    )
    
    # Verificar se os dados foram carregados corretamente
    if X_no_ibge.empty or X_with_ibge.empty:
        print("Erro: Não foi possível carregar os dados.")
        return None
    
    # Treinar modelos para CASOS
    print("\n" + "="*80)
    print("TREINAMENTO PARA CASOS COVID-19")
    print("="*80)
    
    results_cases_no_ibge = train_and_evaluate_model(
        X_no_ibge, y_cases_no_ibge, 
        "Random Forest SEM IBGE", "Casos COVID-19"
    )
    
    results_cases_with_ibge = train_and_evaluate_model(
        X_with_ibge, y_cases_with_ibge, 
        "Random Forest COM IBGE", "Casos COVID-19"
    )
    
    # Treinar modelos para MORTES
    print("\n" + "="*80)
    print("TREINAMENTO PARA MORTES COVID-19")
    print("="*80)
    
    results_deaths_no_ibge = train_and_evaluate_model(
        X_no_ibge, y_deaths_no_ibge, 
        "Random Forest SEM IBGE", "Mortes COVID-19"
    )
    
    results_deaths_with_ibge = train_and_evaluate_model(
        X_with_ibge, y_deaths_with_ibge, 
        "Random Forest COM IBGE", "Mortes COVID-19"
    )
    
    # Comparar resultados
    print("\n" + "="*80)
    print("COMPARACAO DE RESULTADOS")
    print("="*80)
    
    comparison_results = {
        'cases': {
            'no_ibge': results_cases_no_ibge,
            'with_ibge': results_cases_with_ibge
        },
        'deaths': {
            'no_ibge': results_deaths_no_ibge,
            'with_ibge': results_deaths_with_ibge
        }
    }
    
    # Criar tabela comparativa
    create_comparison_table(comparison_results)
    
    # Analisar features IBGE mais importantes
    analyze_ibge_features(results_cases_with_ibge, results_deaths_with_ibge)
    
    return comparison_results

def create_comparison_table(results):
    """Cria tabela comparativa dos resultados"""
    
    print("\n=== TABELA COMPARATIVA DE METRICAS ===")
    
    for target in ['cases', 'deaths']:
        target_name = "CASOS" if target == 'cases' else "MORTES"
        print(f"\n{target_name} COVID-19:")
        print("-" * 60)
        
        no_ibge = results[target]['no_ibge']['test_metrics']
        with_ibge = results[target]['with_ibge']['test_metrics']
        
        print(f"{'Métrica':<20} {'Sem IBGE':<12} {'Com IBGE':<12} {'Melhoria':<10}")
        print("-" * 60)
        
        for metric in no_ibge.keys():
            val_no_ibge = no_ibge[metric]
            val_with_ibge = with_ibge[metric]
            
            if not np.isnan(val_no_ibge) and not np.isnan(val_with_ibge):
                # Para R2 e Explained_Variance, maior é melhor
                if metric in ['R2', 'Explained_Variance']:
                    improvement = ((val_with_ibge - val_no_ibge) / abs(val_no_ibge)) * 100
                    improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
                # Para outras métricas, menor é melhor
                else:
                    improvement = ((val_no_ibge - val_with_ibge) / abs(val_no_ibge)) * 100
                    improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
                
                print(f"{metric:<20} {val_no_ibge:<12.4f} {val_with_ibge:<12.4f} {improvement_str:<10}")
        
        # Tempo de treinamento
        time_no_ibge = results[target]['no_ibge']['training_time']
        time_with_ibge = results[target]['with_ibge']['training_time']
        time_diff = ((time_with_ibge - time_no_ibge) / time_no_ibge) * 100
        time_diff_str = f"+{time_diff:.1f}%" if time_diff > 0 else f"{time_diff:.1f}%"
        
        print(f"{'Tempo Treinamento':<20} {time_no_ibge:<12.2f} {time_with_ibge:<12.2f} {time_diff_str:<10}")

def analyze_ibge_features(cases_results, deaths_results):
    """Analisa as features IBGE mais importantes"""
    
    print("\n=== ANALISE DAS VARIAVEIS IBGE MAIS IMPORTANTES ===")
    
    # Features IBGE para casos
    cases_features = cases_results['feature_importance']
    ibge_cases_features = cases_features[cases_features['feature'].str.startswith('IBGE_')]
    
    # Features IBGE para mortes
    deaths_features = deaths_results['feature_importance']
    ibge_deaths_features = deaths_features[deaths_features['feature'].str.startswith('IBGE_')]
    
    print(f"\nTop 10 variáveis IBGE mais importantes para CASOS:")
    for i, (_, row) in enumerate(ibge_cases_features.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nTop 10 variáveis IBGE mais importantes para MORTES:")
    for i, (_, row) in enumerate(ibge_deaths_features.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Salvar resultados
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Processed')
    os.makedirs(output_dir, exist_ok=True)
    
    ibge_cases_features.to_csv(os.path.join(output_dir, 'ibge_feature_importance_cases.csv'), index=False)
    ibge_deaths_features.to_csv(os.path.join(output_dir, 'ibge_feature_importance_deaths.csv'), index=False)
    
    print(f"\nResultados salvos em:")
    print(f"  - ibge_feature_importance_cases.csv")
    print(f"  - ibge_feature_importance_deaths.csv")

if __name__ == "__main__":
    results = compare_models()
