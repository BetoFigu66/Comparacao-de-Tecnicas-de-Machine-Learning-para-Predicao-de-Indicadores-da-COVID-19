import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import sys
import os
import logging
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from pathlib import Path

# Add the DADOS_COVID directory to Python path
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'DADOS_COVID')) # Path adjusted directly in load_and_prepare_data

def setup_model_logging():
    """Configura o sistema de logging para os modelos"""
    # Criar diretório de logs se não existir
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    log_filename = f"log_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # Também exibe no console
        ]
    )
    
    # Obter logger específico para modelos
    logger = logging.getLogger('models')
    logger.info(f"Sistema de logging iniciado. Arquivo: {log_filepath}")
    
    return logger

# Inicializar logger global para modelos
model_logger = setup_model_logging()

# Optional Keras imports - only needed for neural network models
try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    model_logger.warning("Keras not available. Neural network models will not work.")

# Create your models here.

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Avoid division by zero
    if not mask.any():
        return float('inf')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def load_ibge_features(project_root):
    """Load processed IBGE features for COVID-19 models"""
    
    ibge_file = Path(project_root) / "Data" / "Processed" / "ibge_covid_variables.csv"
    
    if not ibge_file.exists():
        model_logger.warning("IBGE data file not found. Run 'Carga inicial de dados' first to process IBGE data.")
        return None
    
    ibge_data = pd.read_csv(ibge_file)
    model_logger.info(f"Loaded IBGE data: {ibge_data.shape}")
    model_logger.info(f"IBGE variables available: {len([col for col in ibge_data.columns if col.startswith('IBGE_')])}")
    
    return ibge_data

def load_pib_features(project_root):
    """Load processed PIB features for COVID-19 models"""
    
    pib_file = Path(project_root) / "Data" / "Processed" / "pib_covid_variables.csv"
    
    if not pib_file.exists():
        model_logger.warning("PIB data file not found. Run 'Carga inicial de dados' first to process PIB data.")
        return None
    
    pib_data = pd.read_csv(pib_file)
    model_logger.info(f"Loaded PIB data: {pib_data.shape}")
    model_logger.info(f"PIB variables available: {len([col for col in pib_data.columns if col.startswith('PIB_')])}")
    
    return pib_data

def load_synonyms(project_root):
    """Carrega arquivo de sinônimos para correção de nomes de municípios"""
    synonyms_path = os.path.join(project_root, 'Data', 'municipality_synonyms.csv')
    
    if os.path.exists(synonyms_path):
        synonyms_df = pd.read_csv(synonyms_path)
        synonym_map = {}
        for _, row in synonyms_df.iterrows():
            covid_key = normalize_municipality_name(row['covid_name']) + row['covid_uf']
            ibge_key = normalize_municipality_name(row['ibge_name']) + row['ibge_uf']
            synonym_map[covid_key] = ibge_key
        return synonym_map
    return {}

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

def integrate_ibge_with_covid_data(covid_df, ibge_df, project_root):
    """Integrate IBGE features with COVID-19 data using municipality name + UF"""
    
    # Verificar se temos as colunas necessárias
    if 'municipio' not in covid_df.columns or 'uf' not in covid_df.columns:
        model_logger.warning("COVID data must have 'municipio' and 'uf' columns for IBGE integration")
        return covid_df
    
    # Carregar sinônimos
    synonym_map = load_synonyms(project_root)
    
    # Criar chaves de associação
    covid_df['chave_covid'] = covid_df['municipio'].apply(normalize_municipality_name) + covid_df['uf']
    covid_df['chave_covid_final'] = covid_df['chave_covid'].map(lambda x: synonym_map.get(x, x))
    
    # Carregar dados IBGE completos com nomes de municípios
    ibge_excel_path = os.path.join(project_root, 'DataSources', 'IBGE', 'Base_MUNIC_2020.xlsx')
    ibge_raw = pd.read_excel(ibge_excel_path, sheet_name='Recursos humanos')
    
    # Criar chave IBGE
    ibge_raw['chave_ibge'] = ibge_raw['Mun'].apply(normalize_municipality_name) + ibge_raw['Sigla UF']
    
    # Criar mapeamento CodMun -> chave_ibge
    codmun_to_key = dict(zip(ibge_raw['CodMun'], ibge_raw['chave_ibge']))
    
    # Mapear IBGE data para usar chaves
    ibge_df['chave_ibge'] = ibge_df['CodMun'].map(codmun_to_key)
    
    # Fazer merge usando as chaves
    integrated_df = covid_df.merge(
        ibge_df.drop('CodMun', axis=1), 
        left_on='chave_covid_final', 
        right_on='chave_ibge', 
        how='left'
    )
    
    # Remover colunas auxiliares
    integrated_df = integrated_df.drop(['chave_covid', 'chave_covid_final', 'chave_ibge'], axis=1)
    
    # Handle IBGE categorical variables - convert to dummy variables
    ibge_cols = [col for col in integrated_df.columns if col.startswith('IBGE_')]
    categorical_ibge_cols = []
    
    for col in ibge_cols:
        if integrated_df[col].dtype == 'object':
            categorical_ibge_cols.append(col)
    
    if categorical_ibge_cols:
        model_logger.info(f"Converting {len(categorical_ibge_cols)} IBGE categorical variables to dummy variables...")
        integrated_df = pd.get_dummies(
            integrated_df, 
            columns=categorical_ibge_cols, 
            drop_first=True,
            dummy_na=False,  # Não criar coluna para NaN
            dtype='int'
        )
    
    # Show integration results
    new_ibge_features = len([col for col in integrated_df.columns if col.startswith('IBGE_')])
    model_logger.info(f"Integration results:")
    model_logger.info(f"  Original COVID data: {covid_df.shape}")
    model_logger.info(f"  After IBGE integration: {integrated_df.shape}")
    model_logger.info(f"  Total IBGE features (after dummy encoding): {new_ibge_features}")
    
    # Show missing data summary for IBGE features
    ibge_feature_cols = [col for col in integrated_df.columns if col.startswith('IBGE_')]
    if ibge_feature_cols:
        missing_summary = integrated_df[ibge_feature_cols].isnull().sum()
        if missing_summary.sum() > 0:
            model_logger.info(f"  Missing data in IBGE features:")
            for col, missing in missing_summary.items():
                if missing > 0:
                    pct = (missing / len(integrated_df)) * 100
                    model_logger.info(f"    {col}: {missing} ({pct:.1f}%)")
    
    return integrated_df


def normalize_municipality_name_pib(name):
    """Normaliza nomes de municípios para integração PIB (mesmo padrão IBGE)"""
    if pd.isna(name):
        return name
    
    # Converter para string e aplicar normalização
    name = str(name).strip()
    
    # Remover acentos usando a mesma função já importada
    from unidecode import unidecode
    name = unidecode(name)
    
    # Converter para maiúsculas
    name = name.upper()
    
    # Remover apostrofes e underscores
    name = name.replace("'", "").replace("_", "")
    
    return name

def integrate_pib_with_covid_data(covid_data, pib_data, project_root):
    """Integra dados de PIB com dados COVID-19 usando município + UF"""
    
    model_logger.info("Integrating PIB data with COVID-19 data...")
    
    # Normalizar nomes de municípios em ambos datasets
    covid_data_norm = covid_data.copy()
    pib_data_norm = pib_data.copy()
    
    covid_data_norm['municipio_norm'] = covid_data_norm['municipio'].apply(normalize_municipality_name_pib)
    pib_data_norm['municipio_norm'] = pib_data_norm['municipio'].apply(normalize_municipality_name_pib)
    
    # Criar chaves de associação
    covid_data_norm['key'] = covid_data_norm['municipio_norm'] + '_' + covid_data_norm['uf']
    pib_data_norm['key'] = pib_data_norm['municipio_norm'] + '_' + pib_data_norm['uf']
    
    # Carregar sinônimos se existir (mesmo arquivo usado para IBGE)
    synonyms_file = os.path.join(project_root, 'Data', 'municipality_synonyms.csv')
    synonyms = {}
    
    if os.path.exists(synonyms_file):
        try:
            df_synonyms = pd.read_csv(synonyms_file)
            for _, row in df_synonyms.iterrows():
                covid_name_norm = normalize_municipality_name_pib(row['covid_municipality'])
                ibge_name_norm = normalize_municipality_name_pib(row['ibge_municipality'])
                covid_key = covid_name_norm + '_' + row['uf']
                ibge_key = ibge_name_norm + '_' + row['uf']
                synonyms[covid_key] = ibge_key
            model_logger.info(f"Loaded {len(synonyms)} municipality synonyms for PIB integration")
        except Exception as e:
            model_logger.warning(f"Could not load synonyms file: {e}")
    
    # Aplicar sinônimos
    covid_data_norm['key_final'] = covid_data_norm['key'].apply(lambda x: synonyms.get(x, x))
    
    # Fazer merge
    pib_columns = ['PIB_total', 'PIB_per_capita', 'atividade_principal', 'atividade_secundaria', 'atividade_terciaria']
    merge_columns = ['key'] + pib_columns
    
    merged_data = covid_data_norm.merge(
        pib_data_norm[merge_columns], 
        left_on='key_final', 
        right_on='key', 
        how='left',
        suffixes=('', '_pib')
    )
    
    # Remover colunas auxiliares
    columns_to_drop = ['municipio_norm', 'key', 'key_final', 'key_pib']
    merged_data = merged_data.drop(columns=[col for col in columns_to_drop if col in merged_data.columns])
    
    # Verificar sucesso da integração
    total_covid = len(covid_data)
    matched_pib = merged_data['PIB_per_capita'].notna().sum()
    match_rate = (matched_pib / total_covid) * 100
    
    model_logger.info(f"PIB integration results:")
    model_logger.info(f"  Original COVID data: {covid_data.shape}")
    model_logger.info(f"  After PIB integration: {merged_data.shape}")
    model_logger.info(f"  PIB match rate: {match_rate:.2f}% ({matched_pib}/{total_covid})")
    
    # Processar variáveis categóricas de atividade econômica
    activity_columns = ['atividade_principal', 'atividade_secundaria', 'atividade_terciaria']
    
    for col in activity_columns:
        if col in merged_data.columns:
            # Criar dummies para atividades econômicas
            dummies = pd.get_dummies(merged_data[col], prefix=f'PIB_{col}', dummy_na=False)
            merged_data = pd.concat([merged_data, dummies], axis=1)
            # Remover coluna original
            merged_data = merged_data.drop(columns=[col])
    
    # Contar novas features PIB
    pib_features = [col for col in merged_data.columns if col.startswith('PIB_')]
    model_logger.info(f"  Total PIB features added: {len(pib_features)}")
    
    # Tratar valores NaN nas variáveis PIB
    pib_numeric_cols = ['PIB_total', 'PIB_per_capita']
    for col in pib_numeric_cols:
        if col in merged_data.columns:
            # Substituir NaN por 0 ou pela mediana
            median_val = merged_data[col].median()
            merged_data[col] = merged_data[col].fillna(median_val)
            model_logger.info(f"  Filled NaN in {col} with median: {median_val:.2f}")
    
    # Para variáveis dummy de atividade econômica, NaN significa ausência da categoria
    for col in pib_features:
        if col.startswith('PIB_atividade_'):
            merged_data[col] = merged_data[col].fillna(0)
    
    return merged_data

def load_and_prepare_data(uf_target=None, include_ibge=False, include_pib=False):
    # Determine data path based on uf_target
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if uf_target:
        data_path = os.path.join(project_root, 'Data', 'Processed', f'df_final_2020_{uf_target}.parquet')
        model_logger.info(f"Carregando dados específicos para UF: {uf_target} de {data_path}")
    else:
        data_path = os.path.join(project_root, 'Data', 'Processed', 'df_final_2020.parquet')
        model_logger.info(f"Carregando dados gerais de {data_path}")
    
    try:
        dados = pd.read_parquet(data_path)
    except FileNotFoundError:
        model_logger.error(f"Arquivo de dados não encontrado em {data_path}")
        # Return empty DataFrames or raise an error to be handled by the caller
        empty_df = pd.DataFrame()
        return empty_df, pd.Series(dtype='float64'), pd.Series(dtype='float64'), [], None, None
    
    # Load and integrate IBGE municipal variables if requested
    if include_ibge:
        ibge_data = load_ibge_features(project_root)
        if ibge_data is not None:
            # Integrar dados IBGE usando código do município
            dados = dados.merge(ibge_data, left_on='city_ibge_code', right_on='CodMun', how='left')
            dados = dados.drop(columns=['CodMun'], errors='ignore')
            model_logger.info(f"IBGE variables integrated. New dataset shape: {dados.shape}")
        else:
            model_logger.warning("IBGE data not available. Continuing without IBGE features.") 

    # Load and integrate PIB municipal variables if requested
    if include_pib:
        pib_data = load_pib_features(project_root)
        if pib_data is not None:
            # Integrar dados PIB usando código do município
            dados = dados.merge(pib_data, left_on='city_ibge_code', right_on='CodMun', how='left')
            dados = dados.drop(columns=['CodMun'], errors='ignore')
            model_logger.info(f"PIB variables integrated. New dataset shape: {dados.shape}")
        else:
            model_logger.warning("PIB data not available. Continuing without PIB features.")

    df_features_for_segmentation = None # Initialize

    if uf_target:
        # UF-specific files are already dummified for relevant categoricals by carga_dados.py
        # and 'uf' column is absent. No further dummification needed here for those.
        # We might still have other categoricals if the list in carga_dados.py was not exhaustive
        # For now, assume carga_dados.py handled all necessary dummification for UF-specific files.
        pass # No specific dummification here for UF-specific files
    else:
        # General file: dummify 'uf' and other standard categorical columns
        categorical_cols_general = [
            'cor_raca', 'estado_civil', 'faixa_etaria', 'genero',
            'grau_de_instrucao', 'ocupacao', 'partido', 'regiao', 'uf'
        ]
        # Ensure all expected categorical columns exist before trying to dummify
        cols_to_dummify_present = [col for col in categorical_cols_general if col in dados.columns]
        if cols_to_dummify_present:
            dados = pd.get_dummies(
                dados, 
                columns=cols_to_dummify_present, 
                drop_first=False,
                dummy_na=False,  # Não criar coluna para NaN
                dtype='int'
            )
        else:
            model_logger.info("Nenhuma coluna categórica esperada encontrada para dummificar no arquivo geral.")

    # Separate features and targets
    # Ensure 'municipio' column exists before trying to drop it, handle case where it might be missing
    # target_field_deaths = 'last_available_death_rate'
    target_field_deaths = 'death_per_100k_inhabitants'
    target_field_cases = 'last_available_confirmed_per_100k_inhabitants'
    cols_to_drop_from_X = [target_field_cases, target_field_deaths]
    if 'municipio' in dados.columns:
        cols_to_drop_from_X.append('municipio')
    X_original_features = dados.drop(columns=cols_to_drop_from_X)
    y_cases = dados[target_field_cases]
    y_deaths = dados[target_field_deaths]
    
    # LOG DETALHADO DAS VARIÁVEIS PREDITORAS
    model_logger.info(f"\n{'='*60}")
    model_logger.info(f"VARIÁVEIS PREDITORAS PARA TREINAMENTO")
    model_logger.info(f"{'='*60}")
    model_logger.info(f"Total de features: {len(X_original_features.columns)}")
    model_logger.info(f"Registros: {len(X_original_features)}")
    model_logger.info(f"Contexto: {'UF específica: ' + uf_target if uf_target else 'Dados gerais (todas UFs)'}")
    
    # Categorizar e listar as features
    feature_categories = {
        'Eleitorais': [],
        'COVID/População': [],
        'IBGE': [],
        'PIB': [],
        'Geográficas': [],
        'Outras': []
    }
    
    for feature in X_original_features.columns:
        feature_lower = feature.lower()
        if any(term in feature_lower for term in ['partido', 'cor_raca', 'genero', 'ocupacao', 'estado_civil', 'faixa_etaria', 'grau_de_instrucao']):
            feature_categories['Eleitorais'].append(feature)
        elif 'ibge_' in feature_lower:
            feature_categories['IBGE'].append(feature)
        elif 'pib' in feature_lower:
            feature_categories['PIB'].append(feature)
        elif any(term in feature_lower for term in ['population', 'confirmed', 'death']):
            feature_categories['COVID/População'].append(feature)
        elif any(term in feature_lower for term in ['uf_', 'regiao_']):
            feature_categories['Geográficas'].append(feature)
        else:
            feature_categories['Outras'].append(feature)
    
    # Exibir por categoria
    for category, features in feature_categories.items():
        if features:
            model_logger.info(f"\n{category} ({len(features)} features):")
            for i, feature in enumerate(sorted(features), 1):
                model_logger.info(f"  {i:2d}. {feature}")
    
    # Resumo por tipo
    model_logger.info(f"\n{'='*60}")
    model_logger.info(f"RESUMO POR CATEGORIA:")
    for category, features in feature_categories.items():
        if features:
            model_logger.info(f"  {category}: {len(features)} features")
    
    model_logger.info(f"{'='*60}")
    model_logger.info(f"VARIÁVEIS ALVO:")
    model_logger.info(f"  Y_cases: {target_field_cases}")
    model_logger.info(f"  Y_deaths: {target_field_deaths}")
    model_logger.info(f"{'='*60}\n")
    
    if not uf_target: # Only create this for the general dataset
        df_features_for_segmentation = X_original_features.copy()

    # Limpar tipos de dados antes do scaling
    model_logger.info("Limpando tipos de dados antes do scaling...")
    
    for col in X_original_features.columns:
        # Converter valores booleanos string para numéricos
        if X_original_features[col].dtype == 'object':
            # Verificar se são valores booleanos como string
            unique_vals = X_original_features[col].unique()
            if set(str(v).lower() for v in unique_vals if pd.notna(v)).issubset({'true', 'false', '0', '1', '0.0', '1.0'}):
                # Converter True/False string para 1/0
                X_original_features[col] = X_original_features[col].astype(str).map({
                    'True': 1, 'False': 0, 'true': 1, 'false': 0,
                    '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'nan': 0
                }).fillna(0)
                model_logger.info(f"Convertido {col} de booleano string para numérico")
            else:
                # Tentar converter para numérico
                try:
                    X_original_features[col] = pd.to_numeric(X_original_features[col], errors='coerce').fillna(0)
                    model_logger.info(f"Convertido {col} para numérico")
                except:
                    model_logger.warning(f"Não foi possível converter {col}, removendo coluna")
                    X_original_features = X_original_features.drop(columns=[col])
    
    # Garantir que todas as colunas são numéricas
    X_original_features = X_original_features.select_dtypes(include=[np.number])
    model_logger.info(f"Features após limpeza de tipos: {X_original_features.shape}")

    # Scale the features
    X_scaler = StandardScaler()
    X_scaled_array = X_scaler.fit_transform(X_original_features)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=X_original_features.columns)
    
    # Scale the target variables
    y_cases_scaler = StandardScaler()
    y_deaths_scaler = StandardScaler()
    
    y_cases_scaled = y_cases_scaler.fit_transform(y_cases.values.reshape(-1, 1)).ravel()
    y_deaths_scaled = y_deaths_scaler.fit_transform(y_deaths.values.reshape(-1, 1)).ravel()
    
    return X_scaled_df, y_cases_scaled, y_deaths_scaled, X_original_features.columns, (X_scaler, y_cases_scaler, y_deaths_scaler), df_features_for_segmentation

DEFAULT_DT_PARAM_GRID = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def train_decision_tree(X, y, feature_names=None, param_grid=None, model_name=None, data_context=None):
    if param_grid is None:
        param_grid = DEFAULT_DT_PARAM_GRID
    
    # Log das features para este modelo específico
    model_logger.info(f"\n--- TREINANDO {model_name or 'Árvore de Decisão'} ---")
    model_logger.info(f"Features disponíveis: {len(X.columns)}")
    model_logger.info(f"Amostras de treino: {len(X)}")
    model_logger.info(f"Contexto dos dados: {data_context or 'Não especificado'}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tree = DecisionTreeRegressor(random_state=42)
    
    model_logger.info(f"\nIniciando GridSearchCV para {model_name or 'Árvore de Decisão'}")
    model_logger.info(f"Total de combinações de parâmetros: {len(ParameterGrid(param_grid))}")
    
    grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    model_logger.info(f"\nMelhores parâmetros encontrados: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # Calculate metrics
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    model_logger.info(f"R² Score (treino): {train_score:.4f}")
    model_logger.info(f"R² Score (teste): {test_score:.4f}")
    
    # Calculate Permutation Importance
    # X_test is a DataFrame, X_test.columns provides feature names
    perm_importance_result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    # Sort importances in descending order
    sorted_idx = perm_importance_result.importances_mean.argsort()[::-1]
    feature_importances_list = []
    for i in sorted_idx:
        feature_importances_list.append({
            "feature": X_test.columns[i],
            "importance": perm_importance_result.importances_mean[i]
        })
        
    # Prepare return dictionary
    return {
        'model': best_model,
        'train_score': train_score,
        'test_score': test_score,
        'r2_score': test_score,  # For compatibility with the UI
        'best_params': grid_search.best_params_,
        'feature_importances': feature_importances_list,
        'model_name': model_name,
        'data_context': data_context
    }

DEFAULT_SVM_PARAM_GRID = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

def train_svm(X, y, feature_names=None, param_grid=None, model_name=None, data_context=None):
    if param_grid is None:
        param_grid = DEFAULT_SVM_PARAM_GRID
    
    # Log das features para este modelo específico
    model_logger.info(f"\n--- TREINANDO {model_name or 'SVM'} ---")
    model_logger.info(f"Features disponíveis: {len(X.columns)}")
    model_logger.info(f"Amostras de treino: {len(X)}")
    model_logger.info(f"Contexto dos dados: {data_context or 'Não especificado'}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svm = SVR()
    
    model_logger.info(f"\nIniciando GridSearchCV para {model_name or 'Máquina de Vetores de Suporte'}")
    model_logger.info(f"Total de combinações de parâmetros: {len(ParameterGrid(param_grid))}")
    
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    model_logger.info(f"\nMelhores parâmetros encontrados: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # Calculate metrics
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    model_logger.info(f"R² Score (treino): {train_score:.4f}")
    model_logger.info(f"R² Score (teste): {test_score:.4f}")
    
    # Calculate Permutation Importance
    # X_test is a DataFrame, X_test.columns provides feature names
    perm_importance_result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    # Sort importances in descending order
    sorted_idx = perm_importance_result.importances_mean.argsort()[::-1]
    feature_importances_list = []
    for i in sorted_idx:
        feature_importances_list.append({
            "feature": X_test.columns[i],
            "importance": perm_importance_result.importances_mean[i]
        })
        
    # Prepare return dictionary
    return {
        'model': best_model,
        'train_score': train_score,
        'test_score': test_score,
        'r2_score': test_score,  # For compatibility with the UI
        'best_params': grid_search.best_params_,
        'feature_importances': feature_importances_list,
        'model_name': model_name,
        'data_context': data_context
    }

DEFAULT_NN_PARAM_GRID = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}

def train_neural_network(X, y, feature_names=None, param_grid=None, model_name=None, data_context=None):
    if param_grid is None:
        param_grid = DEFAULT_NN_PARAM_GRID
    
    # Log das features para este modelo específico
    model_logger.info(f"\n--- TREINANDO {model_name or 'Rede Neural'} ---")
    model_logger.info(f"Features disponíveis: {len(X.columns)}")
    model_logger.info(f"Amostras de treino: {len(X)}")
    model_logger.info(f"Contexto dos dados: {data_context or 'Não especificado'}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    nn = MLPRegressor(max_iter=1000, random_state=42)
    
    model_logger.info(f"\nIniciando GridSearchCV para {model_name or 'Rede Neural'}")
    model_logger.info(f"Total de combinações de parâmetros: {len(ParameterGrid(param_grid))}")
    
    grid_search = GridSearchCV(nn, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    model_logger.info(f"\nMelhores parâmetros encontrados: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # Calculate metrics
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    model_logger.info(f"R² Score (treino): {train_score:.4f}")
    model_logger.info(f"R² Score (teste): {test_score:.4f}")
    
    # Calculate Permutation Importance
    # X_test is a DataFrame, X_test.columns provides feature names
    perm_importance_result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    # Sort importances in descending order
    sorted_idx = perm_importance_result.importances_mean.argsort()[::-1]
    feature_importances_list = []
    for i in sorted_idx:
        feature_importances_list.append({
            "feature": X_test.columns[i],
            "importance": perm_importance_result.importances_mean[i]
        })
        
    # Prepare return dictionary
    return {
        'model': best_model,
        'train_score': train_score,
        'test_score': test_score,
        'r2_score': test_score,  # For compatibility with the UI
        'best_params': grid_search.best_params_,
        'feature_importances': feature_importances_list,
        'model_name': model_name,
        'data_context': data_context
    }

DEFAULT_RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

def train_random_forest(X, y, feature_names=None, param_grid=None, model_name=None, data_context=None):
    if param_grid is None:
        param_grid = DEFAULT_RF_PARAM_GRID
    
    # Log das features para este modelo específico
    model_logger.info(f"\n--- TREINANDO {model_name or 'Random Forest'} ---")
    model_logger.info(f"Features disponíveis: {len(X.columns)}")
    model_logger.info(f"Amostras de treino: {len(X)}")
    model_logger.info(f"Contexto dos dados: {data_context or 'Não especificado'}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(random_state=42)
    
    model_logger.info(f"\nIniciando GridSearchCV para {model_name or 'Floresta Aleatória'}")
    model_logger.info(f"Total de combinações de parâmetros: {len(ParameterGrid(param_grid))}")
    
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    model_logger.info(f"\nMelhores parâmetros encontrados: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # Calculate metrics
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    model_logger.info(f"R² Score (treino): {train_score:.4f}")
    model_logger.info(f"R² Score (teste): {test_score:.4f}")
    
    # Calculate Permutation Importance
    # X_test is a DataFrame, X_test.columns provides feature names
    perm_importance_result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    # Sort importances in descending order
    sorted_idx = perm_importance_result.importances_mean.argsort()[::-1]
    feature_importances_list = []
    for i in sorted_idx:
        feature_importances_list.append({
            "feature": X_test.columns[i],
            "importance": perm_importance_result.importances_mean[i]
        })
        
    # Prepare return dictionary
    return {
        'model': best_model,
        'train_score': train_score,
        'test_score': test_score,
        'r2_score': test_score,  # For compatibility with the UI
        'best_params': grid_search.best_params_,
        'feature_importances': feature_importances_list,
        'model_name': model_name,
        'data_context': data_context
    }

def train_neural_network_with_dropout(X_train, y_train, input_dim, dropout_rates):
    model = Sequential()
    # Primeira camada oculta com Dropout
    model.add(Dense(30, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rates[0]))  # Dropout após a primeira camada

    # Segunda camada oculta com Dropout
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout_rates[1]))  # Dropout após a segunda camada

    # Camada de saída
    model.add(Dense(1, activation='linear'))

    # Compilando o modelo
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['r2'])

    # Treinando o modelo
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    return model

def train_all_models():
    # Load and prepare data
    X, y_cases, y_deaths, feature_names, scaler = load_and_prepare_data()
    
    models = {}
    
    # Train models for cases prediction
    models['cases'] = {
        'decision_tree': train_decision_tree(X, y_cases),
        'svm': train_svm(X, y_cases),
        'neural_network': train_neural_network_with_dropout(X, y_cases, X.shape[1], [0.2, 0.5]),
        'random_forest': train_random_forest(X, y_cases)
    }
    
    # Train models for death rate prediction
    models['deaths'] = {
        'decision_tree': train_decision_tree(X, y_deaths),
        'svm': train_svm(X, y_deaths),
        'neural_network': train_neural_network_with_dropout(X, y_deaths, X.shape[1], [0.2, 0.5]),
        'random_forest': train_random_forest(X, y_deaths)
    }
    
    return models, feature_names, scaler
