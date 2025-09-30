import os
import sys
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configurar TensorFlow para reduzir mensagens informativas
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduz logs do TensorFlow
import copy
import joblib
import json
import time
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust path to import from parent directory (Scripts) and sibling (Models)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'Models')
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, 'trainedModels')
MODEL_LOG_FILE = os.path.join(TRAINED_MODELS_DIR, "trained_models_log.json")
LOADED_MODEL_INFO = {} # Stores info about the currently loaded model
DATA_LOADED_FLAG = False # Global flag to track if data has been loaded
CURRENT_DATA_CONTEXT = "Nenhum dado carregado" # Indica o contexto dos dados carregados (Geral, UF:SP, etc)

# Variáveis globais para armazenar dados carregados e evitar recarregamento
X_LOADED = None
Y_CASES_LOADED = None
Y_DEATHS_LOADED = None
FEATURE_NAMES_LOADED = None
SCALER_LOADED = None

# Mover configuração de logging para quando for realmente necessário
def setup_logging():
    LOG_DIR = os.path.join(SCRIPT_DIR, 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, f"main_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Criar diretórios sob demanda
def ensure_dirs_exist():
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if SCRIPT_DIR not in sys.path: # To import carga_dados directly
    sys.path.append(SCRIPT_DIR)

# Local imports (after path adjustments)
from Scripts import visualization
from carga_dados import process_and_save_data
from Models.models import (
    load_and_prepare_data,
    train_decision_tree, DEFAULT_DT_PARAM_GRID,
    train_svm, DEFAULT_SVM_PARAM_GRID,
    train_neural_network, DEFAULT_NN_PARAM_GRID,
    train_random_forest, DEFAULT_RF_PARAM_GRID
)

# --- Model Configuration and Help ---
MODEL_TRAINERS = {
    "Árvore de Decisão": ("Decision Tree", train_decision_tree, DEFAULT_DT_PARAM_GRID),
    "SVM (Support Vector Machine)": ("SVM", train_svm, DEFAULT_SVM_PARAM_GRID),
    "Rede Neural (MLP)": ("Neural Network", train_neural_network, DEFAULT_NN_PARAM_GRID),
    "Random Forest": ("Random Forest", train_random_forest, DEFAULT_RF_PARAM_GRID)
}

MODEL_PARAM_HELP = {
    "Árvore de Decisão": {
        "max_depth": "Profundidade máxima da árvore. Valores comuns: 3, 5, 7, 10, None (sem limite). Controla a complexidade.",
        "min_samples_split": "Número mínimo de amostras necessárias para dividir um nó interno. Valores: int (e.g., 2, 5, 10) ou float (fração).",
        "min_samples_leaf": "Número mínimo de amostras necessárias em um nó folha. Valores: int (e.g., 1, 2, 4) ou float (fração)."
    },
    "SVM (Support Vector Machine)": {
        "C": "Parâmetro de regularização. Valores comuns: 0.1, 1, 10. Controla a penalidade por erro de classificação.",
        "kernel": "Tipo de kernel a ser usado. Comuns: 'linear', 'rbf', 'poly'.",
        "gamma": "Coeficiente do kernel para 'rbf', 'poly', 'sigmoid'. Comuns: 'scale', 'auto' ou float."
    },
    "Rede Neural (MLP)": {
        "hidden_layer_sizes": "Tupla especificando o número de neurônios em cada camada oculta. Ex: (50,), (100,), (50,50).",
        "activation": "Função de ativação para as camadas ocultas. Comuns: 'relu', 'tanh', 'logistic'.",
        "alpha": "Parâmetro de regularização L2. Valores: float pequeno, e.g., 0.0001, 0.001, 0.01."
    },
    "Random Forest": {
        "n_estimators": "Número de árvores na floresta. Comuns: 100, 200, 500.",
        "max_depth": "Profundidade máxima de cada árvore. Similar à Árvore de Decisão.",
        "min_samples_split": "Similar à Árvore de Decisão.",
        "min_samples_leaf": "Similar à Árvore de Decisão."
    }
}

# --- Main Functions ---
def handle_data_loading():
    logging.info("Opção selecionada: Carga inicial de dados.")
    global DATA_LOADED_FLAG
    try:
        process_and_save_data()
        logging.info("Script 'carga_dados.py' executado com sucesso.")
        print("✅ Script 'carga_dados.py' executado com sucesso.")
        DATA_LOADED_FLAG = True
    except Exception as e:
        logging.error(f"Erro durante a carga de dados: {e}", exc_info=True)
        print(f"❌ ERRO na carga de dados: {e}")
        print("A carga de dados falhou. Verifique os logs para mais detalhes.")
        DATA_LOADED_FLAG = False

def display_param_grid(model_name, current_grid):
    logging.info(f"Parâmetros atuais para {model_name} (GridSearchCV):")
    for param, values in current_grid.items():
        print(f"  {param}: {values}")

def get_new_param_value(param_name, current_value_list):
    while True:
        print(f"Valores atuais para '{param_name}': {current_value_list}")
        new_values_str = input(f"Digite os novos valores para '{param_name}' separados por vírgula (ou pressione Enter para manter os atuais): ")
        if not new_values_str.strip():
            return current_value_list # Mantém os valores atuais
        try:
            # Tenta converter para o tipo apropriado (int, float, ou manter string)
            # Esta é uma simplificação; uma conversão mais robusta seria necessária
            # baseada no tipo esperado do parâmetro.
            new_values = []
            for val_str in new_values_str.split(','):
                val_str = val_str.strip()

                # Handle empty strings from input like "1,,2"
                if not val_str:
                    logging.debug(f"Valor vazio ('' resultante de split) encontrado. Tratado como string vazia.")
                    new_values.append("") 
                    continue

                # Handle 'None' keyword
                if val_str.lower() == 'none':
                    new_values.append(None)
                    continue
                
                # Attempt integer conversion (common case, no decimal)
                try:
                    new_values.append(int(val_str))
                    continue
                except ValueError:
                    pass # Not an int, try float or treat as string

                # Attempt float conversion
                try:
                    new_values.append(float(val_str))
                    continue
                except ValueError:
                    pass # Not a float, treat as string
                
                # If all conversions fail, append as original string
                logging.debug(f"Valor '{val_str}' não convertido para None/int/float. Mantido como string.")
                new_values.append(val_str)
            return new_values
        except ValueError:
            print("Entrada inválida. Por favor, use o formato correto.")

def show_parameter_help_for_model(model_name):
    logging.info(f"Exibindo ajuda para parâmetros do modelo: {model_name}")
    if model_name in MODEL_PARAM_HELP:
        print(f"\n--- Ajuda para Parâmetros: {model_name} ---")
        for param, desc in MODEL_PARAM_HELP[model_name].items():
            print(f"  {param}: {desc}")
        print("---------------------------------------\n")
    else:
        print("Ajuda não disponível para este modelo.")
    input("\nPressione Enter para continuar...")

def handle_load_preprocessed_data():
    global DATA_LOADED_FLAG, CURRENT_DATA_CONTEXT
    logging.info("Opção selecionada: Carregar dados pré-processados.")
    print("\n--- Carregar Dados Pré-Processados ---")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(base_dir, 'Data', 'Processed')

    while True:
        print("\nEscolha o tipo de dado a carregar:")
        print("1. Carregar dados gerais (todas UFs)")
        print("2. Carregar dados de uma UF específica")
        print("0. Voltar ao menu principal")
        choice = input("Digite sua opção: ")

        actual_uf_target = None
        intended_data_context = ""

        if choice == '1':
            actual_uf_target = None
            intended_data_context = "Geral (todas UFs)"
            print(f"\nCarregando dados gerais...")
            break
        elif choice == '2':
            try:
                files = os.listdir(processed_data_path)
                uf_files = {}
                uf_pattern = re.compile(r"df_final_2020_([A-Z]{2})\.parquet")
                idx = 1
                print("\nUFs disponíveis para carregamento:")
                for f_name in files:
                    match = uf_pattern.match(f_name)
                    if match:
                        uf = match.group(1)
                        uf_files[uf] = uf
                        print(f"{uf}")
                        idx += 1
                
                if not uf_files:
                    print("Nenhum arquivo de UF específica encontrado em Data/Processed/")
                    print("Certifique-se de que a 'Carga inicial de dados' foi executada e gerou os arquivos por UF.")
                    input("\nPressione Enter para continuar...")
                    continue # Volta ao sub-menu de escolha de tipo de dado

                uf_choice = input("Digite o número da UF desejada (ou 0 para cancelar): ")
                if uf_choice in ['0', 'Q', 'q']:
                    continue # Volta ao sub-menu de escolha de tipo de dado
                
                chosen_uf = uf_files.get(uf_choice)
                if chosen_uf:
                    actual_uf_target = chosen_uf
                    intended_data_context = f"UF: {chosen_uf}"
                    print(f"\nCarregando dados para UF: {chosen_uf}...")
                    break
                else:
                    print("Opção de UF inválida.")
                    continue
            except FileNotFoundError:
                print(f"Erro: Diretório de dados processados não encontrado em {processed_data_path}")
                logging.error(f"Diretório de dados processados não encontrado: {processed_data_path}", exc_info=True)
                DATA_LOADED_FLAG = False
                CURRENT_DATA_CONTEXT = "Erro ao acessar diretório de dados"
                input("\nPressione Enter para continuar...")
                return # Volta ao menu principal
            except Exception as e:
                print(f"Erro ao listar arquivos de UF: {e}")
                logging.error(f"Erro ao listar arquivos de UF: {e}", exc_info=True)
                input("\nPressione Enter para continuar...")
                continue # Volta ao sub-menu de escolha de tipo de dado
        elif choice in ['0', 'Q', 'q']:
            return # Volta ao menu principal
        else:
            print("Opção inválida. Tente novamente.")

    # Tentativa de carregar os dados com base na escolha
    try:
        # load_and_prepare_data agora retorna 6 valores
        X, y_cases, y_deaths, feature_names, scaler, _ = load_and_prepare_data(uf_target=actual_uf_target, include_ibge=True, include_pib=True)
        
        if X.empty:
            # load_and_prepare_data já imprimiu o erro de FileNotFoundError
            print(f"Falha ao carregar dados para '{intended_data_context}'. O conjunto de dados está vazio ou não foi encontrado.")
            logging.warning(f"Carga de dados para '{intended_data_context}' resultou em X vazio.")
            DATA_LOADED_FLAG = False
            # CURRENT_DATA_CONTEXT não deve ser alterado para o 'intended' se falhou, mantém o anterior ou indica falha.
            # Para simplificar, vamos redefinir para um estado de falha se a intenção era carregar algo novo.
            if CURRENT_DATA_CONTEXT != intended_data_context: # Evita redefinir se já estava nesse estado de falha
                 CURRENT_DATA_CONTEXT = f"Falha ao carregar: {intended_data_context}"
        else:
            print(f"Dados para '{intended_data_context}' carregados com sucesso. {X.shape[0]} amostras e {X.shape[1]} features.")
            logging.info(f"Dados para '{intended_data_context}' carregados: {X.shape[0]} amostras, {X.shape[1]} features.")
            DATA_LOADED_FLAG = True
            CURRENT_DATA_CONTEXT = intended_data_context # Define o contexto global APÓS sucesso
            
    except Exception as e:
        logging.error(f"Erro ao carregar ou preparar dados para '{intended_data_context}': {e}", exc_info=True)
        print(f"Erro crítico ao carregar dados para '{intended_data_context}': {e}")
        DATA_LOADED_FLAG = False
        CURRENT_DATA_CONTEXT = f"Erro crítico ao carregar: {intended_data_context}"

    input("\nPressione Enter para continuar...")

def handle_load_trained_model():
    global LOADED_MODEL_INFO
    logging.info("Opção selecionada: Carregar modelo treinado.")
    print("\n--- Carregar Modelo Treinado ---")

    if not os.path.exists(MODEL_LOG_FILE) or os.path.getsize(MODEL_LOG_FILE) == 0:
        print(f"Nenhum modelo treinado encontrado no log ({MODEL_LOG_FILE}).")
        logging.warning(f"Arquivo de log de modelos '{MODEL_LOG_FILE}' não encontrado ou vazio.")
        input("Pressione Enter para continuar...")
        return

    try:
        with open(MODEL_LOG_FILE, 'r') as f_log:
            model_logs = json.load(f_log)
    except FileNotFoundError:
        print(f"Arquivo de log de modelos '{MODEL_LOG_FILE}' não encontrado.")
        logging.error(f"Arquivo de log de modelos '{MODEL_LOG_FILE}' não encontrado ao tentar carregar.")
        input("Pressione Enter para continuar...")
        return
    except json.JSONDecodeError:
        print(f"Erro ao ler o arquivo de log de modelos ({MODEL_LOG_FILE}). O arquivo pode estar corrompido.")
        logging.error(f"Erro de decodificação JSON em '{MODEL_LOG_FILE}'.")
        input("Pressione Enter para continuar...")
        return

    if not model_logs:
        print("Nenhum modelo registrado no log.")
        input("Pressione Enter para continuar...")
        return

    print("Modelos disponíveis para carregar:")
    for i, entry in enumerate(model_logs):
        print(f"  {i+1}. {entry.get('model_name', 'N/A')} (Alvo: {entry.get('target_variable', 'N/A')}, Data: {entry.get('timestamp', 'N/A')})")
    print("  0. Retornar ao menu principal")

    while True:
        choice_str = input("Escolha o número do modelo para carregar (ou 0 para retornar): ")
        try:
            if choice_str in ['0', 'Q', 'q']:
                logging.info("Carregamento de modelo cancelado pelo usuário.")
                return
            if 1 <= int(choice_str) <= len(model_logs):
                selected_entry = model_logs[int(choice_str) - 1]
                model_path = selected_entry.get('saved_model_path')
                if not model_path or not os.path.exists(model_path):
                    print(f"Erro: Arquivo do modelo '{model_path}' não encontrado. Verifique o log.")
                    logging.error(f"Arquivo do modelo '{model_path}' listado no log não foi encontrado.")
                    input("Pressione Enter para continuar...")
                    return
                
                try:
                    loaded_model = joblib.load(model_path)
                    LOADED_MODEL_INFO = {
                        "model_name": selected_entry.get('model_name'),
                        "target_variable": selected_entry.get('target_variable'),
                        "timestamp": selected_entry.get('timestamp'),
                        "model_path": model_path,
                        "model_object": loaded_model,
                        "best_parameters": selected_entry.get('best_parameters_found'),
                        "feature_importances": selected_entry.get('feature_importances'),
                        "training_parameters_grid": selected_entry.get('training_parameters_grid'),
                        "train_score_r2": selected_entry.get('train_score_r2'),
                        "test_score_r2": selected_entry.get('test_score_r2'),
                        "data_context": selected_entry.get('data_context', 'N/A') # Added for consistency
                    }
                    print(f"Modelo '{LOADED_MODEL_INFO['model_name']}' (Contexto: {LOADED_MODEL_INFO.get('data_context', 'N/A')}) carregado com sucesso.")
                    logging.info(f"Modelo '{LOADED_MODEL_INFO['model_name']}' (path: {model_path}, contexto: {LOADED_MODEL_INFO.get('data_context', 'N/A')}) carregado.")
                    input("Pressione Enter para continuar...")
                    return
                except Exception as e: # This except was part of the original try for joblib.load
                    print(f"Erro ao carregar o arquivo do modelo '{model_path}': {e}")
                    logging.error(f"Erro ao carregar o modelo de '{model_path}': {e}", exc_info=True)
                    input("Pressione Enter para continuar...")
                    return # Return from handle_load_trained_model
            else: # This else was part of the if 1 <= choice_num <= len(model_logs)
                print("Opção inválida. Por favor, escolha um número da lista.")
        except ValueError: # This except was part of the try for int(choice_str)
            print("Entrada inválida. Por favor, digite um número.")

def handle_model_training():
    global DATA_LOADED_FLAG, CURRENT_DATA_CONTEXT # Ensure CURRENT_DATA_CONTEXT is accessible
    logging.info("Opção selecionada: Treinamento de modelo.")

    if not DATA_LOADED_FLAG:
        print("\nOs dados precisam ser carregados antes de treinar um modelo.")
        print("Por favor, use a opção '2. Carregar dados pré-processados' no menu principal.")
        logging.warning("Tentativa de treinar modelo sem dados carregados (DATA_LOADED_FLAG=False).")
        input("Pressione Enter para continuar...")
        return
    print("\nQual modelo você gostaria de treinar?")
    option = 1
    for key, (name, _, _) in MODEL_TRAINERS.items():
        print(f"  {option}. {name}")
        option += 1
    print("  0. Retornar ao menu principal")
    
    model_choice = input("Escolha uma opção: ")

    if model_choice in ['0', 'Q', 'q']:
        logging.info("Seleção de modelo cancelada. Retornando ao menu principal.")
        return

    try:
        model_choice_num = int(model_choice)
        if model_choice_num < 1 or model_choice_num > len(MODEL_TRAINERS):
            raise IndexError("Opção fora do intervalo válido")
        
        # Obter o modelo pela posição na lista
        model_items = list(MODEL_TRAINERS.items())
        model_key, (model_name, model_trainer_func, default_param_grid) = model_items[model_choice_num-1]
        
    except (ValueError, IndexError):
        print("Opção inválida.")
        logging.warning(f"Tentativa de treinar modelo com opção inválida: {model_choice}")
        return
    current_param_grid = copy.deepcopy(default_param_grid) # Trabalha com uma cópia

    # The old UF segmentation prompt is removed as this is now handled by CURRENT_DATA_CONTEXT

    while True:
        print(f"\n--- Configurando {model_name} ---")
        display_param_grid(model_name, current_param_grid)
        print("\nOpções de configuração:")
        param_keys = list(current_param_grid.keys())
        for i, p_key in enumerate(param_keys):
            print(f"  {i+1}. Modificar '{p_key}'")
        help_option_num = len(param_keys) + 1
        train_option_num = len(param_keys) + 2
        print(f"  {help_option_num}. Ajuda para parâmetros de {model_name}")
        print(f"  {train_option_num}. Iniciar Treinamento com parâmetros atuais")
        print(f"  0. Voltar ao menu principal")

        config_choice = input("Escolha uma opção de configuração: ")
        
        try:
            help_option_num = len(param_keys) + 1
            train_option_num = len(param_keys) + 2

            if config_choice in ['0', 'Q', 'q']:
                logging.info(f"Configuração de {model_name} cancelada. Retornando ao menu principal.")
                break # Sai do loop de configuração, retorna ao menu principal
            elif 1 <= int(config_choice) <= len(param_keys):
                param_to_modify = param_keys[int(config_choice)-1]
                new_values = get_new_param_value(param_to_modify, current_param_grid[param_to_modify])
                current_param_grid[param_to_modify] = new_values
                logging.info(f"Parâmetro '{param_to_modify}' para {model_name} atualizado para: {new_values}")
            elif int(config_choice) == help_option_num:
                show_parameter_help_for_model(model_name)
            elif int(config_choice) == train_option_num:
                logging.info(f"Iniciando treinamento para {model_name} com grid: {current_param_grid} e contexto de dados: {CURRENT_DATA_CONTEXT}")
                print(f"Carregando dados ({CURRENT_DATA_CONTEXT}) para treinamento...")

                parsed_uf_target = None
                data_context_for_load = CURRENT_DATA_CONTEXT # Use a local copy for decisions within this training attempt

                if data_context_for_load.startswith("UF: "):
                    try:
                        parsed_uf_target = data_context_for_load.split(": ")[1]
                    except IndexError:
                        logging.warning(f"Não foi possível parsear UF do CURRENT_DATA_CONTEXT: {data_context_for_load}. Usando dados gerais.")
                        parsed_uf_target = None 
                        data_context_for_load = "Geral (todas UFs)" # Fallback context
                elif data_context_for_load == "Geral (todas UFs)":
                    parsed_uf_target = None
                else: 
                    print(f"Alerta: Contexto de dados '{data_context_for_load}' é inválido ou indica falha. Tentando carregar dados gerais como fallback.")
                    logging.warning(f"Contexto de dados '{data_context_for_load}' inválido/falha. Fallback para dados gerais.")
                    parsed_uf_target = None
                    data_context_for_load = "Geral (todas UFs)" # Fallback context

                try:
                    # load_and_prepare_data agora retorna 6 valores, o último é df_features_for_segmentation (não usado aqui)
                    X, y_cases, y_deaths, feature_names, scaler, _ = load_and_prepare_data(uf_target=parsed_uf_target, include_ibge=True, include_pib=True)
                    
                    if X.empty:
                        print(f"Falha ao carregar dados para o contexto '{data_context_for_load}'. O treinamento não pode prosseguir.")
                        logging.error(f"Treinamento de {model_name} abortado: X está vazio para contexto {data_context_for_load}.")
                        break # Sai do loop de configuração e volta ao menu principal
                    # Decidir qual target usar (ex: y_cases por padrão, ou perguntar ao usuário)
                    # Para este exemplo, vamos usar y_cases
                    print(f"Treinando {model_name} para predição de casos...")
                    # A função de treino específica (ex: train_decision_tree) usa GridSearchCV internamente
                    start_time = time.time()
                    # model_trainer_func retorna um dicionário com todas as informações
                    result = model_trainer_func(X, y_cases, param_grid=current_param_grid)
                    end_time = time.time()
                    training_duration = end_time - start_time
                    
                    # Extrair valores do dicionário retornado
                    best_model = result['model']
                    train_score = result['train_score']
                    test_score = result['test_score']
                    best_params = result['best_params']
                    feature_importances = result['feature_importances']
                    logging.info(f"Treinamento de {model_name} (casos) concluído em {training_duration:.2f} segundos.")
                    logging.info(f"  Melhores Parâmetros: {best_params}")
                    logging.info(f"  Score de Treino (R^2): {train_score:.4f}")
                    logging.info(f"  Score de Teste (R^2): {test_score:.4f}")
                    print(f"Treinamento de {model_name} (casos) concluído.")
                    print(f"  Melhores Parâmetros: {best_params}")
                    print(f"  Score de Treino (R^2): {train_score:.4f}")
                    print(f"  Score de Teste (R^2): {test_score:.4f}")

                    # --- Save Model and Log --- 
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    target_variable_name = "cases" # Hardcoded for now, can be made dynamic
                    
                    # Determine data_context_suffix based on the actual context used for loading data
                    actual_data_context_suffix = "_Geral"
                    if data_context_for_load.startswith("UF: "):
                        uf_val = data_context_for_load.split(": ")[1]
                        actual_data_context_suffix = f"_UF_{uf_val}"
                    elif data_context_for_load == "Geral (todas UFs)":
                        actual_data_context_suffix = "_Geral"
                    # else: it remains "_Geral" due to fallback logic above

                    model_filename = f"{model_name.replace(' ', '_')}_{target_variable_name}{actual_data_context_suffix}_{timestamp_str}.joblib"
                    model_save_path = os.path.join(TRAINED_MODELS_DIR, model_filename)

                    try:
                        joblib.dump(best_model, model_save_path)
                        logging.info(f"Modelo '{model_name}' salvo em: {model_save_path}")
                        print(f"Modelo salvo em: {model_save_path}")

                        # Prepare feature importances for logging, ensuring it's not None
                        feature_importances_to_log = []
                        if feature_importances is not None:
                            feature_importances_to_log = feature_importances[:20]
                        else:
                            logging.critical(f"'feature_importances' was None for model {model_name} before logging. Using empty list.")

                        # Calcular métricas adicionais
                        y_pred = best_model.predict(X_test if 'X_test' in locals() else X.iloc[-int(len(X)*0.2):])
                        y_test_vals = y_cases.iloc[-int(len(y_cases)*0.2):] if 'y_test' not in locals() else y_test
                        
                        from sklearn.metrics import mean_squared_error, mean_absolute_error
                        import numpy as np
                        
                        mse = mean_squared_error(y_test_vals, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test_vals, y_pred)
                        
                        # MAPE
                        def calculate_mape(y_true, y_pred):
                            y_true, y_pred = np.array(y_true), np.array(y_pred)
                            mask = y_true != 0
                            if not mask.any():
                                return float('inf')
                            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                        
                        mape = calculate_mape(y_test_vals, y_pred)

                        model_log_entry = {
                            "model_name": model_name,
                            "target_variable": target_variable_name,
                            "timestamp": timestamp_str,
                            "training_parameters_grid": current_param_grid, # Grid used for GridSearchCV
                            "best_parameters_found": best_params,      # Best params found by GridSearchCV
                            "train_score_r2": train_score,
                            "test_score_r2": test_score,
                            "mse": float(mse),
                            "rmse": float(rmse),
                            "mae": float(mae),
                            "mape": float(mape),
                            "training_duration_seconds": training_duration,
                            "feature_importances": feature_importances_to_log, # Save top 20 features (or empty list if None)
                            "saved_model_path": model_save_path, # Store full path for now, can be relative
                            "data_context": data_context_for_load # Log the actual data context used for this training run
                        }

                        model_logs = []
                        if os.path.exists(MODEL_LOG_FILE) and os.path.getsize(MODEL_LOG_FILE) > 0:
                            with open(MODEL_LOG_FILE, 'r') as f:
                                try:
                                    model_logs = json.load(f)
                                except json.JSONDecodeError:
                                    logging.error(f"Erro ao decodificar {MODEL_LOG_FILE}. Iniciando com lista vazia.")
                                    model_logs = [] # Reset if file is corrupt
                        
                        model_logs.append(model_log_entry)

                        with open(MODEL_LOG_FILE, 'w') as f:
                            json.dump(model_logs, f, indent=4)
                        logging.info(f"Metadados do modelo adicionados a {MODEL_LOG_FILE}")

                        # Print top 10 features
                        print("\n--- Importância das Variáveis (Permutation Importance) ---")
                        if feature_importances:
                            for i, item in enumerate(feature_importances[:10]): # Display top 10
                                print(f"  {i+1}. {item['feature']}: {item['importance']:.4f}")
                        else:
                            print("  Informação de importância de variáveis não disponível.")
                        print("--------------------------------------------------------")

                    except Exception as save_exc:
                        logging.error(f"Erro ao salvar o modelo treinado ou seus metadados: {save_exc}", exc_info=True)
                        print(f"Erro ao salvar o modelo: {save_exc}")
                        input("Pressione Enter para continuar...")
                        
                except Exception as e:
                    print(f"Erro durante o carregamento de dados ou treinamento de {model_name}: {e}")
                    logging.error(f"Erro durante o carregamento de dados ou treinamento de {model_name}: {e}", exc_info=True)
                    print("Erro no treinamento:", e)
                    input("Pressione Enter para continuar...")
                    break # Sai do loop de configuração, volta ao menu principal após o treinamento
            else:
                print("Opção de configuração inválida.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

def handle_train_all_models():
    """Treina todos os modelos automaticamente (Decision Tree, SVM, Neural Network, Random Forest)"""
    global DATA_LOADED_FLAG, CURRENT_DATA_CONTEXT
    logging.info("Opção selecionada: Treinar todos os modelos (automático).")
    
    if not DATA_LOADED_FLAG:
        print("❌ Erro: Dados não carregados. Carregue os dados primeiro.")
        input("\nPressione Enter para continuar...")
        return
    
    print("\n--- TREINAMENTO AUTOMÁTICO DE TODOS OS MODELOS ---")
    print(f"Contexto dos dados: {CURRENT_DATA_CONTEXT}")
    
    # Lista de todos os modelos disponíveis
    all_models = [
        ("Decision Tree", "decision_tree"),
        ("SVM", "svm"), 
        ("Neural Network", "neural_network"),
        ("Random Forest", "random_forest")
    ]
    
    # Lista de variáveis alvo
    targets = [
        ("Casos COVID-19", "cases"),
        ("Mortes COVID-19", "deaths")
    ]
    
    print(f"\nSerão treinados {len(all_models)} modelos para {len(targets)} variáveis alvo.")
    print(f"Total de treinamentos: {len(all_models) * len(targets)}")
    
    confirm = input("\nDeseja continuar? (s/n): ").lower().strip()
    if confirm != 's':
        print("Treinamento cancelado.")
        input("\nPressione Enter para continuar...")
        return
    
    successful_trainings = 0
    failed_trainings = 0
    
    # Treinar todos os modelos para todas as variáveis alvo
    for target_name, target_key in targets:
        print(f"\n{'='*60}")
        print(f"TREINANDO MODELOS PARA: {target_name}")
        print(f"{'='*60}")
        
        for model_name, model_key in all_models:
            try:
                print(f"\n🔄 Treinando {model_name} para {target_name}...")
                
                # Carregar dados
                uf_target = CURRENT_DATA_CONTEXT if CURRENT_DATA_CONTEXT != "Geral" else None
                X, y_cases, y_deaths, feature_names, scaler, _ = load_and_prepare_data(uf_target=uf_target, include_ibge=True, include_pib=True)
                
                # Selecionar variável alvo
                y = y_cases if target_key == "cases" else y_deaths
                
                # Configurar nome do modelo
                context_suffix = f"_{CURRENT_DATA_CONTEXT}" if CURRENT_DATA_CONTEXT != "Geral" else ""
                full_model_name = f"{model_name}_{target_name.replace(' ', '_')}{context_suffix}"
                
                # Treinar modelo
                if model_key == "decision_tree":
                    model_info = train_decision_tree(X, y, feature_names, model_name=full_model_name, data_context=CURRENT_DATA_CONTEXT)
                elif model_key == "svm":
                    model_info = train_svm(X, y, feature_names, model_name=full_model_name, data_context=CURRENT_DATA_CONTEXT)
                elif model_key == "neural_network":
                    model_info = train_neural_network(X, y, feature_names, model_name=full_model_name, data_context=CURRENT_DATA_CONTEXT)
                elif model_key == "random_forest":
                    model_info = train_random_forest(X, y, feature_names, model_name=full_model_name, data_context=CURRENT_DATA_CONTEXT)
                
                # Salvar modelo
                try:
                    save_trained_model(model_info, full_model_name, target_key, CURRENT_DATA_CONTEXT)
                    print(f"✅ {model_name} treinado e salvo com sucesso!")
                    successful_trainings += 1
                except Exception as save_exc:
                    logging.error(f"Erro ao salvar {model_name}: {save_exc}")
                    print(f"❌ Erro ao salvar {model_name}: {save_exc}")
                    failed_trainings += 1
                    
            except Exception as e:
                logging.error(f"Erro no treinamento de {model_name} para {target_name}: {e}")
                print(f"❌ Erro no treinamento de {model_name}: {e}")
                failed_trainings += 1
    
    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO DO TREINAMENTO AUTOMÁTICO")
    print(f"{'='*60}")
    print(f"✅ Treinamentos bem-sucedidos: {successful_trainings}")
    print(f"❌ Treinamentos falharam: {failed_trainings}")
    print(f"📊 Total de treinamentos: {successful_trainings + failed_trainings}")
    
    if successful_trainings > 0:
        print(f"\n🎉 Treinamento automático concluído!")
        print(f"Os modelos foram salvos e podem ser carregados posteriormente.")
    
    input("\nPressione Enter para continuar...")

def handle_train_models_from_config():
    """Treina múltiplos modelos baseado nas configurações do arquivo models_to_train.json"""
    config_file = os.path.join(SCRIPT_DIR, 'models_to_train.json')
    if not os.path.exists(config_file):
        logging.error("Arquivo de configuração models_to_train.json não encontrado")
        print("Erro: Arquivo de configuração não encontrado")
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Erro ao ler arquivo de configuração: {e}")
        print("Erro: Arquivo de configuração inválido")
        return

    models = config.get('models', [])
    if not models:
        print("Nenhum modelo configurado para treinamento")
        return

    print(f"\nEncontrados {len(models)} modelos para treinar...")
    
    for i, model_config in enumerate(models, 1):
        # Verificar se já existe um modelo com os mesmos parâmetros
        modelo_existente = None
        if os.path.exists(MODEL_LOG_FILE):
            try:
                with open(MODEL_LOG_FILE, 'r', encoding='utf-8') as f:
                    modelos_log = json.load(f)
                
                # Garantir que modelos_log é uma lista válida
                if not isinstance(modelos_log, list):
                    modelos_log = []
                    logging.warning("O arquivo de log de modelos não contém uma lista válida")
                    
                # Verificar cada modelo no log
                for modelo in modelos_log:
                    # Verificar se o tipo, nome, contexto e variável alvo são os mesmos
                    if (modelo.get('model_type') == model_type and 
                        modelo.get('model_name') == model_config.get('name') and
                        modelo.get('data_context') == (f"UF: {uf_target}" if uf_target else "Geral") and
                        modelo.get('target_variable') == model_config.get('target', 'cases')):
                        
                        # Verificar se os parâmetros são os mesmos
                        best_params = modelo.get('best_parameters_found', {})
                        params_match = True
                        
                        # Comparar parâmetros do grid com os melhores parâmetros encontrados
                        for param_name, param_values in param_grid.items():
                            if param_name in best_params:
                                # Se o valor ótimo não está no grid, não é o mesmo modelo
                                if best_params[param_name] not in param_values:
                                    params_match = False
                                    break
                            else:
                                # Se um parâmetro está faltando, não é o mesmo modelo
                                params_match = False
                                break
                        
                        if params_match:
                            modelo_existente = modelo
                            break
            except Exception as e:
                logging.error(f"Erro ao verificar modelos existentes: {e}")
                # Continuar com o treinamento mesmo se falhar a verificação
        
        # Se encontrou modelo existente, pular treinamento
        if modelo_existente:
            print(f"\nModelo já existe com os mesmos parâmetros!")
            print(f"Nome: {modelo_existente.get('model_name')}")
            print(f"Treinado em: {modelo_existente.get('timestamp')}")
            print(f"R² Score: {modelo_existente.get('test_score_r2')}")
            print(f"Caminho: {modelo_existente.get('saved_model_path')}")
            continue
        
        current_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        print(f"\n[{current_time}] Treinando modelo {i}/{len(models)}:")
        print(f"Tipo: {model_config.get('type')}")
        print(f"Nome: {model_config.get('name')}")
        
        # Configurar contexto de dados
        uf_target = model_config.get('uf')
        data_context = f"UF: {uf_target}" if uf_target else "Geral"
        
        if uf_target:
            print(f"UF: {uf_target}")
        else:
            print("Contexto: Dados Gerais")
        
        # Variáveis para armazenar os dados carregados
        global CURRENT_DATA_CONTEXT, DATA_LOADED_FLAG
        global X_LOADED, Y_CASES_LOADED, Y_DEATHS_LOADED, FEATURE_NAMES_LOADED, SCALER_LOADED
        
        # Verificar se os dados já estão carregados com o mesmo contexto
        if DATA_LOADED_FLAG and CURRENT_DATA_CONTEXT == data_context:
            print(f"Reutilizando dados já carregados para contexto: {data_context}")
            X = X_LOADED
            y_cases = Y_CASES_LOADED
            y_deaths = Y_DEATHS_LOADED
            feature_names = FEATURE_NAMES_LOADED
            scaler = SCALER_LOADED
        else:
            # Carregar novos dados
            try:
                print(f"Carregando novos dados para contexto: {data_context}")
                X, y_cases, y_deaths, feature_names, scaler, _ = load_and_prepare_data(uf_target=uf_target, include_ibge=True, include_pib=True)
                
                # Armazenar os dados carregados para possível reutilização
                X_LOADED = X
                Y_CASES_LOADED = y_cases
                Y_DEATHS_LOADED = y_deaths
                FEATURE_NAMES_LOADED = feature_names
                SCALER_LOADED = scaler
                DATA_LOADED_FLAG = True
                CURRENT_DATA_CONTEXT = data_context
                
                print("Dados carregados com sucesso")
            except Exception as e:
                logging.error(f"Erro ao carregar dados para modelo {model_config.get('name')}: {e}")
                print(f"Erro ao carregar dados: {e}")
                continue

        # Selecionar função de treinamento baseado no tipo
        model_type = model_config.get('type')
        
        if model_type not in MODEL_TRAINERS:
            print(f"Tipo de modelo desconhecido: {model_type}")
            print("Tipos válidos: Árvore de Decisão, SVM (Support Vector Machine), Rede Neural (MLP), Random Forest")
            continue

        _, trainer_func, default_params = MODEL_TRAINERS[model_type]
        
        # Preparar os parâmetros para grid search
        config_params = model_config.get('parameters', {})
        param_grid = {}
        
        # Para cada parâmetro no config, criar uma lista para grid search
        for param_name, param_value in config_params.items():
            # Se o valor já é uma lista, usar diretamente
            if isinstance(param_value, list):
                param_grid[param_name] = param_value
            # Se não é uma lista, criar uma lista com um único valor
            else:
                param_grid[param_name] = [param_value]

        # Treinar modelo se não existir
        try:
            print("Iniciando treinamento...")
            # Registrar tempo de início do treinamento
            start_time = datetime.now()
            
            model_info = trainer_func(
                X, y_cases if model_config.get('target') == 'cases' else y_deaths,
                feature_names,
                param_grid=param_grid,
                model_name=model_config.get('name'),
                data_context=f"UF: {uf_target}" if uf_target else "Geral"
            )
            
            # Calcular tempo de treinamento em segundos
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            model_info['training_time'] = training_time
            
            print(f"Modelo {model_config.get('name')} treinado com sucesso!")
            print(f"R² Score: {model_info.get('test_score', 'N/A')}")
            print(f"Tempo de treinamento: {training_time:.2f} segundos")
            
            # Salvar modelo após treinamento bem-sucedido
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{model_info['model_name'].replace(' ', '_')}_{timestamp}.joblib"
            model_path = os.path.join(TRAINED_MODELS_DIR, model_filename)
            
            try:
                # Verificar se o modelo está disponível no dicionário model_info
                if 'model' in model_info:
                    joblib.dump(model_info['model'], model_path)
                    print(f"Modelo salvo em: {model_path}")
                else:
                    print(f"Erro: Modelo não disponível no resultado do treinamento")
                    logging.error(f"Modelo não disponível no resultado do treinamento para {model_info['model_name']}")
                    continue
                
                # Ordenar feature_importances por valor absoluto (módulo) e limitar a 20 features
                feature_importances = model_info.get('feature_importances', [])
                if feature_importances:
                    # Filtrar features com importância diferente de zero
                    non_zero_importances = [f for f in feature_importances if f.get('importance', 0) != 0]
                    
                    # Ordenar por valor absoluto (módulo) da importância, em ordem decrescente
                    sorted_importances = sorted(non_zero_importances, key=lambda x: abs(x.get('importance', 0)), reverse=True)
                    
                    # Limitar a 20 features (se houver mais de 20 não-zero)
                    sorted_importances = sorted_importances[:20]
                    
                    print(f"Features com importância não-zero: {len(non_zero_importances)} de {len(feature_importances)}")
                else:
                    sorted_importances = []
                
                # Atualizar log de modelos
                model_log_entry = {
                    'model_name': model_info['model_name'],
                    'model_type': model_type,
                    'target_variable': model_config.get('target', 'cases'),
                    'timestamp': timestamp,
                    'saved_model_path': model_path,
                    'best_parameters_found': model_info.get('best_parameters'),
                    'feature_importances': sorted_importances,
                    'train_score_r2': model_info.get('train_score'),
                    'test_score_r2': model_info.get('test_score'),
                    'data_context': model_info.get('data_context'),
                    'training_time': model_info.get('training_time', 0)
                }
                
                # Atualizar arquivo de log
                if os.path.exists(MODEL_LOG_FILE):
                    try:
                        with open(MODEL_LOG_FILE, 'r', encoding='utf-8') as f:
                            model_log = json.load(f)
                    except json.JSONDecodeError:
                        model_log = []
                else:
                    model_log = []
                    
                model_log.append(model_log_entry)
                
                with open(MODEL_LOG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(model_log, f, indent=4)
                
            except Exception as e:
                print(f"Erro ao salvar modelo: {e}")
                logging.error(f"Erro ao salvar modelo {model_info['model_name']}: {e}")
                # Continuar mesmo se falhar ao salvar
                
        except Exception as e:
            logging.error(f"Erro ao treinar modelo {model_config.get('name')}: {e}")
            print(f"Erro no treinamento: {e}")
            continue

    print("\nTreinamento em lote concluído!")
    input("\nPressione Enter para continuar...")

def handle_show_loaded_model_data():
    global LOADED_MODEL_INFO
    logging.info("Opção selecionada: Mostrar dados do modelo carregado.")
    print("\n--- Dados do Modelo Carregado ---")

    if LOADED_MODEL_INFO is None or not LOADED_MODEL_INFO.get('model_object'):
        print("Nenhum modelo está carregado atualmente.")
        logging.info("Nenhum modelo carregado para mostrar dados.")
        input("Pressione Enter para continuar...")
        return

    print(f"  Nome do Modelo: {LOADED_MODEL_INFO.get('model_name', 'N/A')}")
    print(f"  Variável Alvo: {LOADED_MODEL_INFO.get('target_variable', 'N/A')}")
    print(f"  Timestamp do Treinamento: {LOADED_MODEL_INFO.get('timestamp', 'N/A')}")
    print(f"  Caminho do Modelo Salvo: {LOADED_MODEL_INFO.get('model_path', 'N/A')}")
    
    print("\n  Melhores Parâmetros Encontrados:")
    best_params = LOADED_MODEL_INFO.get('best_parameters')
    if best_params and isinstance(best_params, dict):
        for param, value in best_params.items():
            print(f"    {param}: {value}")
    else:
        print("    N/A")

    print("\n  Scores R²:")
    train_score_val = LOADED_MODEL_INFO.get('train_score_r2', 'N/A')
    test_score_val = LOADED_MODEL_INFO.get('test_score_r2', 'N/A')
    
    train_score_str = f"{train_score_val:.4f}" if isinstance(train_score_val, (int, float)) else str(train_score_val)
    test_score_str = f"{test_score_val:.4f}" if isinstance(test_score_val, (int, float)) else str(test_score_val)
    
    print(f"    Score de Treino (R²): {train_score_str}")
    print(f"    Score de Teste (R²): {test_score_str}")

    print("\n  Importância das Variáveis (Top Features):")
    feature_importances = LOADED_MODEL_INFO.get('feature_importances')
    if feature_importances and isinstance(feature_importances, list):
        if not feature_importances: # Check if the list is empty
            print("    Nenhuma informação de importância de característica disponível.")
        for i, item in enumerate(feature_importances): # Already top 20 (or fewer)
            print(f"    {i+1}. {item.get('feature', 'N/A')}: {item.get('importance', 0.0):.4f}")
    else:
        print("    N/A ou formato inválido.")

    # Optionally, display training_parameters_grid if needed, could be verbose
    # print("\n  Grid de Parâmetros Usado no Treinamento:")
    # training_grid = LOADED_MODEL_INFO.get('training_parameters_grid')
    # if training_grid and isinstance(training_grid, dict):
    #     for param, value in training_grid.items():
    #         print(f"    {param}: {value}")
    # else:
    #     print("    N/A")

    input("\nPressione Enter para continuar...")

def handle_generate_visualizations():
    """Gera visualizações a partir dos modelos treinados."""
    logging.info("Opção selecionada: Gerar visualizações e gráficos")
    print("\n--- GERAÇÃO DE GRÁFICOS ---")
    
    # Verificar se o arquivo de log existe
    if not os.path.exists(MODEL_LOG_FILE):
        print(f"Arquivo de log {MODEL_LOG_FILE} não encontrado.")
        logging.error(f"Arquivo de log {MODEL_LOG_FILE} não encontrado")
        input("\nPressione Enter para continuar...")
        return
    
    # Listar os arquivos json do diretório TRAINED_MODELS_DIR
    log_files = [f for f in os.listdir(TRAINED_MODELS_DIR) if f.endswith('.json')]
    if not log_files:
        print(f"Nenhum arquivo json encontrado em {TRAINED_MODELS_DIR}.")
        logging.error(f"Nenhum arquivo json encontrado em {TRAINED_MODELS_DIR}.")
        input("\nPressione Enter para continuar...")
        return
    
    print("\nEscolha o arquivo de log para gerar os gráficos:")
    for i, log_file in enumerate(log_files, 1):
        print(f"{i}. {log_file}")
    log_choice = input(f"\nEscolha uma opção (padrão: 1): ") or "1"
    log_file = log_files[int(log_choice) - 1]
    
    if log_choice == "1":
        log_file = MODEL_LOG_FILE
    else:
        log_file = os.path.join(TRAINED_MODELS_DIR, "trained_models_log_resorted.json")
        if not os.path.exists(log_file):
            print(f"Arquivo {log_file} não encontrado. Usando o arquivo padrão.")
            log_file = MODEL_LOG_FILE
    
    # Perguntar onde salvar os gráficos
    output_dir = os.path.join(PROJECT_ROOT, "Visualizacoes")
    # custom_dir = input(f"\nDigite o diretório para salvar os gráficos (padrão: {output_dir}): ")
    # if custom_dir:
    #     output_dir = custom_dir
    
    # Garantir que o diretório exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Perguntar ao usuário quantas features exibir para os gráficos de importância
    max_features = 10  # Valor padrão
    try:
        max_features_input = input(f"\nNúmero máximo de features a exibir por modelo (padrão: {max_features}): ").strip()
        if max_features_input:
            max_features = int(max_features_input)
    except ValueError:
        print(f"Valor inválido. Usando o padrão: {max_features}")
        

    # Gerar os gráficos
    print(f"\nGerando gráficos a partir de {log_file}...")
    try:
        visualization.generate_all_plots(log_file, output_dir, max_features)
        print(f"\nGráficos gerados com sucesso no diretório: {output_dir}")
    except Exception as e:
        logging.error(f"Erro ao gerar gráficos: {e}")
        print(f"Erro ao gerar gráficos: {e}")
    
    input("\nPressione Enter para continuar...")

# ========================================
# FUNÇÕES DE ANÁLISE DE MODELOS
# ========================================

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

def handle_update_model_metrics():
    """Atualiza as métricas dos modelos já treinados"""
    
    print("=== ATUALIZAR MÉTRICAS DE MODELOS ===")
    print("Adicionando MSE, RMSE, MAE e MAPE aos modelos já treinados...")
    print()
    
    # Caminhos
    trained_models_dir = os.path.join(PROJECT_ROOT, 'Models', 'trainedModels')
    log_file = os.path.join(trained_models_dir, 'trained_models_log.json')
    
    if not os.path.exists(log_file):
        print(f"[ERRO] Arquivo {log_file} não encontrado!")
        input("\nPressione Enter para continuar...")
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
    
    input("\nPressione Enter para continuar...")

def handle_generate_feature_importance_csv():
    """Gera CSV com feature importance de todos os modelos"""
    
    print("=== GERAR CSV FEATURE IMPORTANCE ===")
    print("Criando matriz: Features x Modelos com importâncias...")
    print()
    
    # Caminhos
    trained_models_dir = os.path.join(PROJECT_ROOT, 'Models', 'trainedModels')
    log_file = os.path.join(trained_models_dir, 'trained_models_log.json')
    
    if not os.path.exists(log_file):
        print(f"[ERRO] Arquivo {log_file} não encontrado!")
        input("\nPressione Enter para continuar...")
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
        input("\nPressione Enter para continuar...")
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
    
    input("\nPressione Enter para continuar...")

def handle_list_predictor_variables():
    """Lista todas as variáveis preditoras categorizadas"""
    data_path = os.path.join(PROCESSED_DATA_DIR, 'df_final_2020.parquet')
    
    if not os.path.exists(data_path):
        print("❌ Arquivo de dados não encontrado. Execute a carga inicial primeiro.")
        return
    
    try:
        # Carregar dados
        print("[LOAD] Carregando dados do parquet...")
        df = pd.read_parquet(data_path)
        
        print(f"[INFO] Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
        
        # Separar variáveis por categoria
        all_columns = df.columns.tolist()
        
        # Remover variáveis target, identificadores e as especificadas pelo usuário
        exclude_vars = ['cases_per_100k', 'deaths_per_100k', 'city_ibge_code', 'municipio', 'uf', 
                       'death_per_100k_inhabitants', 'estimated_population', 'last_available_confirmed_per_100k_inhabitants']
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
            if var.startswith('IBGE_'):
                categories['IBGE'].append(var)
            elif var.startswith('PIB_') or 'pib' in var.lower():
                categories['PIB'].append(var)
            elif any(term in var.lower() for term in ['partido', 'eleicao', 'votos', 'candidato', 'prefeito']):
                categories['Eleitorais'].append(var)
            elif any(term in var.lower() for term in ['confirmed', 'death', 'cases', 'population']):
                categories['COVID/População'].append(var)
            elif any(term in var.lower() for term in ['regiao', 'uf', 'municipio', 'codigo']):
                categories['Geográficas'].append(var)
            else:
                categories['Outras'].append(var)
        
        # Salvar lista completa
        output_file = os.path.join(PROCESSED_DATA_DIR, f'predictor_variables_{datetime.now().strftime("%Y%m%d_%H%M")}.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("LISTA COMPLETA DE VARIÁVEIS PREDITORAS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas\n")
            f.write(f"Total de variáveis preditoras: {len(predictor_vars)}\n\n")
            
            for category, vars_list in categories.items():
                if vars_list:
                    f.write(f"{category.upper()} ({len(vars_list)} variáveis):\n")
                    f.write("-" * 30 + "\n")
                    for i, var in enumerate(sorted(vars_list), 1):
                        f.write(f"{i:3d}. {var}\n")
                    f.write("\n")
            
            f.write("\nVARIÁVEIS EXCLUÍDAS:\n")
            f.write("-" * 20 + "\n")
            for var in exclude_vars:
                if var in all_columns:
                    f.write(f"- {var}\n")
        
        # Mostrar resumo na tela
        print("📊 RESUMO DAS VARIÁVEIS PREDITORAS:")
        print("=" * 40)
        for category, vars_list in categories.items():
            if vars_list:
                print(f"{category}: {len(vars_list)} variáveis")
        
        print(f"\n📁 Lista completa salva em: {output_file}")
        
    except Exception as e:
        print(f"❌ Erro ao listar variáveis: {e}")
        import traceback
        traceback.print_exc()

def handle_correlation_analysis():
    """Analisa correlações entre variáveis preditoras e permite remoção de variáveis altamente correlacionadas"""
    data_path = os.path.join(PROCESSED_DATA_DIR, 'df_final_2020.parquet')
    
    if not os.path.exists(data_path):
        print("❌ Arquivo de dados não encontrado. Execute a carga inicial primeiro.")
        return
    
    try:
        # Carregar dados
        print("[LOAD] Carregando dados do parquet...")
        df = pd.read_parquet(data_path)
        
        print(f"[INFO] Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
        
        # Obter variáveis preditoras numéricas
        exclude_vars = ['cases_per_100k', 'deaths_per_100k', 'city_ibge_code', 'municipio', 'uf', 
                       'death_per_100k_inhabitants', 'estimated_population', 'last_available_confirmed_per_100k_inhabitants']
        
        # Selecionar apenas colunas numéricas para correlação
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        predictor_vars = [col for col in numeric_cols if col not in exclude_vars]
        
        if len(predictor_vars) < 2:
            print("❌ Não há variáveis numéricas suficientes para análise de correlação.")
            return
        
        print(f"[INFO] Analisando correlações entre {len(predictor_vars)} variáveis numéricas...")
        
        # Calcular matriz de correlação
        df_predictors = df[predictor_vars]
        correlation_matrix = df_predictors.corr()
        
        # Encontrar correlações perfeitas (1.0)
        perfect_correlations = []
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) == 1.0:
                    perfect_correlations.append((var1, var2, corr_value))
                elif abs(corr_value) > 0.9:
                    high_correlations.append((var1, var2, corr_value))
        
        # Mostrar resultados
        print("\n" + "="*80)
        print("📊 ANÁLISE DE CORRELAÇÃO ENTRE VARIÁVEIS PREDITORAS")
        print("="*80)
        
        # Correlações perfeitas (1.0)
        if perfect_correlations:
            print(f"\n🎯 CORRELAÇÕES PERFEITAS (|r| = 1.0): {len(perfect_correlations)} encontradas")
            print("-" * 60)
            for i, (var1, var2, corr) in enumerate(perfect_correlations, 1):
                print(f"{i:2d}. {var1}")
                print(f"    ↔ {var2}")
                print(f"    Correlação: {corr:.3f}")
                print()
        else:
            print("\n✅ Nenhuma correlação perfeita encontrada.")
        
        # Correlações altas (> 0.9)
        if high_correlations:
            print(f"\n⚠️ CORRELAÇÕES ALTAS (|r| > 0.9): {len(high_correlations)} encontradas")
            print("-" * 60)
            for i, (var1, var2, corr) in enumerate(high_correlations, 1):
                print(f"{i:2d}. {var1}")
                print(f"    ↔ {var2}")
                print(f"    Correlação: {corr:.3f}")
                print()
        else:
            print("\n✅ Nenhuma correlação alta (>0.9) encontrada.")
        
        # Salvar relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_file = os.path.join(PROCESSED_DATA_DIR, f'correlation_analysis_{timestamp}.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ANÁLISE DE CORRELAÇÃO ENTRE VARIÁVEIS PREDITORAS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas\n")
            f.write(f"Variáveis numéricas analisadas: {len(predictor_vars)}\n\n")
            
            f.write(f"CORRELAÇÕES PERFEITAS (|r| = 1.0): {len(perfect_correlations)}\n")
            f.write("-" * 40 + "\n")
            for i, (var1, var2, corr) in enumerate(perfect_correlations, 1):
                f.write(f"{i:2d}. {var1} ↔ {var2} (r = {corr:.3f})\n")
            
            f.write(f"\nCORRELAÇÕES ALTAS (|r| > 0.9): {len(high_correlations)}\n")
            f.write("-" * 40 + "\n")
            for i, (var1, var2, corr) in enumerate(high_correlations, 1):
                f.write(f"{i:2d}. {var1} ↔ {var2} (r = {corr:.3f})\n")
        
        print(f"\n📁 Relatório salvo em: {report_file}")
        
        # Opção interativa para remover variáveis
        if perfect_correlations or high_correlations:
            print("\n" + "="*60)
            print("🛠️ REMOÇÃO INTERATIVA DE VARIÁVEIS CORRELACIONADAS")
            print("="*60)
            
            all_correlations = perfect_correlations + high_correlations
            variables_to_remove = []
            
            for i, (var1, var2, corr) in enumerate(all_correlations, 1):
                print(f"\n{i}. Correlação: {corr:.3f}")
                print(f"   Variável 1: {var1}")
                print(f"   Variável 2: {var2}")
                
                while True:
                    choice = input("   Qual remover? (1/2/n=nenhuma/s=sair): ").strip().lower()
                    if choice == '1':
                        if var1 not in variables_to_remove:
                            variables_to_remove.append(var1)
                            print(f"   ✅ {var1} marcada para remoção")
                        break
                    elif choice == '2':
                        if var2 not in variables_to_remove:
                            variables_to_remove.append(var2)
                            print(f"   ✅ {var2} marcada para remoção")
                        break
                    elif choice == 'n':
                        print("   ⏭️ Nenhuma variável removida")
                        break
                    elif choice == 's':
                        print("   🛑 Saindo da seleção")
                        break
                    else:
                        print("   ❌ Opção inválida. Use 1, 2, n ou s")
                
                if choice == 's':
                    break
            
            # Aplicar remoções se houver
            if variables_to_remove:
                print(f"\n📋 RESUMO: {len(variables_to_remove)} variáveis marcadas para remoção:")
                for var in variables_to_remove:
                    print(f"   - {var}")
                
                confirm = input("\n❓ Confirma a remoção? (s/n): ").strip().lower()
                if confirm == 's':
                    # Criar novo dataset sem as variáveis removidas
                    df_cleaned = df.drop(columns=variables_to_remove)
                    
                    # Salvar dataset limpo
                    cleaned_file = os.path.join(PROCESSED_DATA_DIR, f'df_final_2020_cleaned_{timestamp}.parquet')
                    df_cleaned.to_parquet(cleaned_file, index=False)
                    
                    print(f"\n✅ Dataset limpo salvo: {cleaned_file}")
                    print(f"   Original: {df.shape[1]} colunas")
                    print(f"   Limpo: {df_cleaned.shape[1]} colunas")
                    print(f"   Removidas: {len(variables_to_remove)} colunas")
                    
                    # Salvar lista de variáveis removidas
                    removed_file = os.path.join(PROCESSED_DATA_DIR, f'removed_variables_{timestamp}.txt')
                    with open(removed_file, 'w', encoding='utf-8') as f:
                        f.write("VARIÁVEIS REMOVIDAS POR ALTA CORRELAÇÃO\n")
                        f.write("="*50 + "\n\n")
                        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Total removidas: {len(variables_to_remove)}\n\n")
                        for i, var in enumerate(variables_to_remove, 1):
                            f.write(f"{i:2d}. {var}\n")
                    
                    print(f"📁 Lista de variáveis removidas: {removed_file}")
                else:
                    print("❌ Remoção cancelada")
            else:
                print("\n📝 Nenhuma variável selecionada para remoção")
        
    except Exception as e:
        print(f"❌ Erro na análise de correlação: {e}")
        import traceback
        traceback.print_exc()

def main_menu():
    global LOADED_MODEL_INFO # DATA_LOADED_FLAG and CURRENT_DATA_CONTEXT are module globals

    # Define all potential menu options with their conditions and handlers
    all_potential_options = [
        {"text": "Carga inicial de dados (gera o arquivo .parquet)", 
         "handler": handle_data_loading, 
         "condition": lambda: True},
        {"text": "Carregar dados pré-processados (Geral ou por UF)", 
         "handler": handle_load_preprocessed_data, 
         "condition": lambda: True},
        {"text": "Treinamento de modelo individual", 
         "handler": handle_model_training, 
         "condition": lambda: DATA_LOADED_FLAG},
        # {"text": "Treinar todos os modelos (automático)",
        #  "handler": handle_train_all_models,
        #  "condition": lambda: DATA_LOADED_FLAG},
        {"text": "Treinamento em lote (models_to_train.json)",
         "handler": handle_train_models_from_config,
         "condition": lambda: True},
        {"text": "Carregar modelo treinado", 
         "handler": handle_load_trained_model, 
         "condition": lambda: True},
        {"text": "Mostrar dados do modelo carregado", 
         "handler": handle_show_loaded_model_data, 
         "condition": lambda: bool(LOADED_MODEL_INFO)},
        {"text": "Gerar visualizações e gráficos", 
         "handler": handle_generate_visualizations, 
         "condition": lambda: True},
        {"text": "Atualizar métricas de modelos (MSE, RMSE, MAE, MAPE)", 
         "handler": handle_update_model_metrics, 
         "condition": lambda: True},
        {"text": "Gerar CSV com Feature Importance por modelo", 
         "handler": handle_generate_feature_importance_csv, 
         "condition": lambda: True},
        {"text": "Listar todas as variáveis preditoras", 
         "handler": handle_list_predictor_variables, 
         "condition": lambda: True},
        {"text": "Análise de correlação entre variáveis (remoção interativa)", 
         "handler": handle_correlation_analysis, 
         "condition": lambda: True}
    ]

    logging.info("Aplicação iniciada V01.")
    while True:
        print("\n---------------------------------------------")
        print("   COVID-19 INDICATORS PREDICTION - CLI")
        if LOADED_MODEL_INFO:
            model_display_name = LOADED_MODEL_INFO.get('model_name', 'N/A')
            target_var = LOADED_MODEL_INFO.get('target_variable', 'N/A')
            timestamp = LOADED_MODEL_INFO.get('timestamp', 'N/A')
            data_ctx = LOADED_MODEL_INFO.get('data_context', 'N/A')
            print(f"   Modelo Carregado: {model_display_name} (Alvo: {target_var}, Timestamp: {timestamp}, Contexto: {data_ctx})")
        print("---------------------------------------------")
        print("--- MENU PRINCIPAL ---")
        print("=" * 22)
        print(f"   Contexto Atual dos Dados: {CURRENT_DATA_CONTEXT}")
        print("-" * 22)

        # Filter and number available options
        current_menu_options = []
        for option_config in all_potential_options:
            if option_config["condition"]():
                current_menu_options.append(option_config)
        
        for i, option_config in enumerate(current_menu_options):
            print(f"{i+1}. {option_config['text']}")
        
        print("0. Sair")
        choice = input("Escolha uma opção: ")

        if choice in ['0', 'Q', 'q']:
            logging.info("Aplicação encerrada.")
            print("Saindo...")
            break
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(current_menu_options):
                selected_option_config = current_menu_options[choice_num - 1]
                selected_option_config["handler"]()
            else:
                logging.warning(f"Opção numérica fora do intervalo no menu principal: {choice_num}")
                print("Opção inválida. Tente novamente.")
        except ValueError:
            logging.warning(f"Opção não numérica inválida no menu principal: {choice}")
            print("Opção inválida. Por favor, insira um número.")

if __name__ == "__main__":
    ensure_dirs_exist()  # Garante que os diretórios existam
    setup_logging()      # Configura logging apenas quando necessário
    main_menu()
    # handle_train_models_from_config()
    #visualization.generate_all_plots(MODEL_LOG_FILE, "Visualizacoes", 10)
