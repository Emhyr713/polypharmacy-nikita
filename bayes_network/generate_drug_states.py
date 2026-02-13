# generate_drug_states.py

import json
from collections import defaultdict
import os
from datetime import datetime

# --- Вспомогательная функция логирования (для этого скрипта) ---
def log_to_file(filename, message, add_timestamp=True):
    """
    Добавляет сообщение в лог-файл.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if add_timestamp else ""
    prefix = f"[{timestamp_str}] " if add_timestamp else ""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"{prefix}{message}\n")

# --- Функция загрузки JSON (для этого скрипта) ---
def load_json(filename):
    """Загружает JSON из файла."""
    # log_to_file("generate_log.txt", f"Loading JSON from file: {filename}") # Можно раскомментировать для отладки
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # log_to_file("generate_log.txt", f"Successfully loaded JSON from: {filename}") # Можно раскомментировать для отладки
    return data

def create_drug_states_file(graph_data, output_file='drug_states.json', default_state=0):
    """
    Создает файл drug_states.json со всеми узлами с 'label': 'prepare',
    инициализированными с default_state.
    """
    log_to_file("generate_log.txt", f"Generating drug states file: {output_file}")
    drug_states = {}
    for node in graph_data['nodes']:
        if node.get('label') == 'prepare':
            drug_states[node['id']] = default_state
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(drug_states, f, indent=4, ensure_ascii=False)
    log_to_file("generate_log.txt", f"Successfully created {output_file}.")

if __name__ == "__main__":
    graph_path = 'bayes_network\\data\\graphs\\graphs_10_5.json'
    drug_states_output_path = 'bayes_network\\data\\drug_states.json'
    log_file_for_generator = 'bayes_network\\data\\generate_log.txt'

    # Очистка лога генератора перед каждым новым запуском
    if os.path.exists(log_file_for_generator):
        os.remove(log_file_for_generator)
    log_to_file(log_file_for_generator, "--- Drug States Generator Started ---")

    try:
        if not os.path.exists(graph_path):
            log_to_file(log_file_for_generator, f"ERROR: Graph file '{graph_path}' not found. Cannot generate drug_states.json.")
            raise FileNotFoundError(f"Файл '{graph_path}' не найден. Пожалуйста, создайте его.")

        graph_data = load_json(graph_path)
        create_drug_states_file(graph_data, drug_states_output_path)
        print(f"Файл '{drug_states_output_path}' успешно создан на основе '{graph_path}'.")
        print(f"Вы можете отредактировать '{drug_states_output_path}', а затем запустить 'optimize_bayesian_network.py'.")
        log_to_file(log_file_for_generator, "--- Drug States Generator Completed ---")

    except Exception as e:
        print(f"Произошла ошибка при генерации drug_states.json: {e}")
        log_to_file(log_file_for_generator, f"FATAL ERROR: {e}")