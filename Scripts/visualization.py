import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_model_log(log_file):
    """Load and parse the model training log file."""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Model log file not found: {log_file}")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        try:
            log_data = json.load(f)
            
            # Verificar se o log_data é uma lista (formato atual) ou um dicionário com chave 'models'
            if isinstance(log_data, list):
                return log_data
            elif isinstance(log_data, dict) and 'models' in log_data:
                return log_data.get('models', [])
            else:
                print(f"Formato de arquivo de log não reconhecido: {log_file}")
                return []
        except json.JSONDecodeError:
            print(f"Erro ao decodificar o arquivo de log: {log_file}")
            return []

def plot_r2_comparison(models_data, output_dir):
    """Create a bar plot comparing R² scores of different models."""
    # Extract model names and R² scores
    model_names = []
    train_r2 = []
    test_r2 = []
    
    for model in models_data:
        # Adaptar para o formato atual do log
        name = model.get('model_name', 'Desconhecido')
        context = model.get('data_context', '')
        if context:
            name = f"{name} ({context})"
            
        model_names.append(name)
        train_r2.append(model.get('train_score_r2', 0))
        test_r2.append(model.get('test_score_r2', 0))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Model': model_names,
        'Train R²': train_r2,
        'Test R²': test_r2
    })
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    df.plot(x='Model', y=['Train R²', 'Test R²'], kind='bar', width=0.8)
    plt.title('Comparação de R² entre Modelos')
    plt.xlabel('Modelo')
    plt.ylabel('R²')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'r2_comparison.png')
    plt.savefig(output_path)
    plt.close()
    return output_path

def plot_training_time(models_data, output_dir):
    """Create a bar plot comparing training times of different models."""
    model_names = []
    training_times = []
    
    for model in models_data:
        # Adaptar para o formato atual do log
        name = model.get('model_name', 'Desconhecido')
        context = model.get('data_context', '')
        if context:
            name = f"{name} ({context})"
            
        # Obter o tempo de treinamento
        training_time = model.get('training_time', 0)
        
        # Adicionar apenas se o tempo for maior que zero
        if training_time > 0:
            model_names.append(name)
            training_times.append(training_time)
    
    # Se não houver dados válidos, retornar
    if not model_names:
        print("Nenhum dado de tempo de treinamento válido encontrado.")
        return None
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, training_times)
    plt.title('Tempo de Treinamento por Modelo')
    plt.xlabel('Modelo')
    plt.ylabel('Tempo (segundos)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_time.png')
    plt.savefig(output_path)
    plt.close()
    return output_path

def plot_feature_importance(models_data, output_dir, max_features):
    """Create plots comparing feature importance across models.
    
    Args:
        models_data: Lista de dicionários com dados dos modelos
        output_dir: Diretório onde os gráficos serão salvos
        max_features: Número máximo de features a serem exibidas por modelo
    """
    # Collect feature importance data
    feature_importance_data = []
    
    for model in models_data:
        feature_importances = model.get('feature_importances', [])
        if feature_importances:
            model_name = model.get('model_name', 'Desconhecido')
            context = model.get('data_context', '')
            if context:
                model_name = f"{model_name} ({context})"
            
            # Extrair features e importâncias
            features = [item.get('feature', '') for item in feature_importances]
            importances = [item.get('importance', 0.0) for item in feature_importances]
            
            # Convert to DataFrame
            model_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances,
                'Model': model_name
            })
            feature_importance_data.append(model_df)
    
    if not feature_importance_data:
        return None
    
    # Combine all data
    all_importance = pd.concat(feature_importance_data, ignore_index=True)
    
    # Verificar se há combinações duplicadas de Model e Feature
    duplicates = all_importance.duplicated(subset=['Model', 'Feature'])
    if duplicates.any():
        print(f"Aviso: Encontradas {duplicates.sum()} combinações duplicadas de modelo e feature.")
        # Agrupar por Model e Feature e calcular a média das importâncias
        all_importance = all_importance.groupby(['Model', 'Feature'], as_index=False)['Importance'].mean()
    
    try:
        # Calcular a importância total de cada feature somando os valores absolutos em todos os modelos
        feature_total_importance = all_importance.groupby('Feature')['Importance'].apply(lambda x: abs(x).sum()).reset_index()
        feature_total_importance = feature_total_importance.sort_values('Importance', ascending=False)
        
        # Selecionar apenas as 15 features mais importantes
        top_15_features = feature_total_importance.head(15)['Feature'].tolist()
        
        # Filtrar o DataFrame original para incluir apenas as top 15 features
        filtered_importance = all_importance[all_importance['Feature'].isin(top_15_features)]
        
        # Create a heatmap com apenas as top 15 features
        plt.figure(figsize=(15, 8))
        pivot_table = filtered_importance.pivot(index='Model', columns='Feature', values='Importance')
        sns.heatmap(pivot_table, annot=True, cmap='RdYlBu', center=0, fmt='.2f')
        plt.title('Importância das 15 Features Mais Relevantes por Modelo')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    except Exception as e:
        print(f"Erro ao criar heatmap: {e}")
        # Criar um gráfico alternativo se o heatmap falhar
        plt.figure(figsize=(15, 8))
        plt.text(0.5, 0.5, f"Não foi possível criar o heatmap: {e}", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
    
    output_path = os.path.join(output_dir, 'feature_importance_heatmap.png')
    plt.savefig(output_path)
    plt.close()
    
    # Create individual bar plots for each model
    for model_name in all_importance['Model'].unique():
        model_data = all_importance[all_importance['Model'] == model_name].copy()
        
        # Ordenar pelo valor absoluto da importância (módulo)
        model_data['AbsImportance'] = model_data['Importance'].abs()
        model_data = model_data.sort_values('AbsImportance', ascending=False)
        
        # Limitar ao número máximo de features especificado
        top_features = model_data.head(max_features)
        
        # Inverter a ordem para que as features com maiores valores absolutos fiquem no topo
        top_features = top_features.iloc[::-1]
        
        plt.figure(figsize=(10, 6))
        
        # Criar barras com cores diferentes para valores positivos e negativos
        colors = ['#2196F3' if x >= 0 else '#FF5252' for x in top_features['Importance']]
        bars = plt.barh(top_features['Feature'], top_features['Importance'], color=colors)
        
        min_width = min(top_features['Importance'])
        # Adicionar valores nas barras com posicionamento melhorado para evitar sobreposição
        for bar in bars:
            width = bar.get_width()
            if width >= 0:
                # Para valores positivos, coloca o texto à direita da barra
                label_x_pos = width + 0.01
                ha = 'left'
            else:
                # Para valores negativos, coloca o texto à esquerda da barra
                label_x_pos = width - 0.01
                if (label_x_pos < min_width+0.01):
                    label_x_pos = min_width+0.01
                ha = 'right'
            
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', va='center', ha=ha, fontsize=9)
        
        plt.title(f'Top {max_features} Features por Importância - {model_name}')
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)  # Linha vertical no zero
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # plt.xlim(0.01-min_width, None)  # Deslocar a area de plotagem um pouco para a direita
        #plt.xlim(plt.xlim()[0]-0.01, plt.xlim()[1])  # Deslocar a area de plotagem um pouco para a direita
        print(model_name)
        print('xlim=', plt.xlim())
        print('min_width=', min_width)
        
        model_filename = f"feature_importance_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        output_path = os.path.join(output_dir, model_filename)
        plt.savefig(output_path)
        plt.close()
    
    return output_path

def plot_feature_count(models_data, output_dir):
    """Create a bar plot showing the number of non-zero importance features for each model."""
    model_names = []
    feature_counts = []
    
    for model in models_data:
        # Adaptar para o formato atual do log
        name = model.get('model_name', 'Desconhecido')
        context = model.get('data_context', '')
        if context:
            name = f"{name} ({context})"
        
        # Contar o número de features com importância não-zero
        feature_importances = model.get('feature_importances', [])
        if feature_importances:
            feature_counts.append(len(feature_importances))
            model_names.append(name)
    
    # Se não houver dados válidos, retornar
    if not model_names:
        print("Nenhum dado de importância de features encontrado.")
        return None
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, feature_counts)
    plt.title('Quantidade de Features com Importância Não-Zero por Modelo')
    plt.xlabel('Modelo')
    plt.ylabel('Número de Features')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Adicionar os valores acima das barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_count.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_all_plots(log_file, output_dir='plots', max_features=5):
    """Generate all visualization plots from the model log file.
    
    Args:
        log_file: Caminho para o arquivo de log
        output_dir: Diretório onde os gráficos serão salvos
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model data
        models_data = load_model_log(log_file)
        
        if not models_data:
            print("Nenhum dado de modelo encontrado no arquivo de log.")
            return
        
        # Generate plots
        r2_plot = plot_r2_comparison(models_data, output_dir)
        time_plot = plot_training_time(models_data, output_dir)
        feature_count_plot = plot_feature_count(models_data, output_dir)
        
        # Gerar gráficos de importância de features
        importance_plots = plot_feature_importance(models_data, output_dir, max_features)
        
        print(f"\nGráficos gerados com sucesso no diretório: {output_dir}")
        print(f"- Comparação de R²: {r2_plot}")
        print(f"- Tempo de Treinamento: {time_plot}")
        print(f"- Quantidade de Features: {feature_count_plot}")
        if importance_plots:
            print(f"- Importância das Variáveis: {importance_plots}")
        
    except Exception as e:
        print(f"Erro ao gerar gráficos: {str(e)}")
