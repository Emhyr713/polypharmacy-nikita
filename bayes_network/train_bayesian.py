"""
Оптимизация вероятностей для байесовской сети
С проверкой всех препаратов из sef_dataset.json
"""

import json
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from datetime import datetime
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# Импорт функций из модуля coolBNet
from coolBNet import (load_graph,
                      build_network,
                      calculate_probabilities,
                      load_combined_data,
                    #   BayesianNode,
                      create_base_probabilities
                      )

# Константы
# FREQUENCY_TO_PROB = {
#     "очень часто": 0.55,
#     "часто": 0.055,
#     "нечасто": 0.0055,
#     "редко": 0.00055,
#     "очень редко": 0.00055,
#     "частота неизвестна": 0.000055,
# }

FREQUENCY_TO_PROB = {
    "очень часто": 0.7,
    "часто": 0.5,
    "нечасто": 0.2,
    "редко": 0.1,
    "очень редко": 0.05,
    "частота неизвестна": 0.05,
}

WORK_DIR = "bayes_network"

# Конфигурационные файлы
GRAPH_FILE = f'{WORK_DIR}\\spironolacton_lizinopril.json'
TARGET_FILE = f'{WORK_DIR}\\sef_dataset.json'
OUTPUT_FILE = f'{WORK_DIR}\\probabilities_opt.json'
LOG_FILE = f'{WORK_DIR}\\opt_calc.txt'
PROB_FILE = f'{WORK_DIR}\\probabilities.json'
TRACE_FILE = f'{WORK_DIR}\\calculation_trace.txt'

def init_log():
    """Инициализация файла лога"""
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"=== Лог оптимизации {datetime.now()} ===\n\n")

def log(message):
    """Запись сообщения в лог"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def load_target_effects():
    """Загрузка целевых эффектов из файла"""
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    effects = {}
    for drug, info in data.items():
        effects[drug] = {
            effect: FREQUENCY_TO_PROB.get(freq, 0.0)
            for effect, freq in info['side_e_parts'].items()
        }
    return effects

class ProbabilityOptimizer:
    """Класс для оптимизации вероятностей"""
    
    def __init__(self, graph, target_effects):
        self.graph = graph
        self.target_effects = target_effects
        self.drug_map = self._find_all_drug_effects()

        if os.path.exists(PROB_FILE):
            self.initial_probs = load_combined_data(prob_file=PROB_FILE)
        else:
            self.initial_probs = create_base_probabilities(graph)
        
        self._prepare_optimization_data()
        self.bounds = [(0.00001, 0.9999999) for _ in self.initial_params]
    
    def _find_all_drug_effects(self):
        """Находит все препараты и их эффекты в графе"""
        drug_effects = defaultdict(list)
        missing_drugs = []
        missing_effects = defaultdict(list)
        
        log("\nПоиск препаратов и эффектов в графе:")
        
        # Проверяем все препараты из target_effects
        for drug_name in self.target_effects.keys():
            # Ищем узел препарата (без учета регистра)
            drug_node = next(
                (n for n in self.graph['nodes'] 
                 if n.get('name', '').lower() == drug_name.lower()), 
                None
            )
            
            if not drug_node:
                missing_drugs.append(drug_name)
                continue
            
            # Ищем эффекты для этого препарата
            found_effects = 0
            for effect_name in self.target_effects[drug_name].keys():
                effect_node = next(
                    (n for n in self.graph['nodes'] 
                     if n.get('name', '').lower() == effect_name.lower()), 
                    None
                )
                
                if effect_node:
                    drug_effects[drug_name].append((effect_node['id'], effect_name))
                    found_effects += 1
                else:
                    missing_effects[drug_name].append(effect_name)
            
            log(f"Препарат '{drug_name}': найдено {found_effects} эффектов")
        
        # Логируем отсутствующие препараты
        if missing_drugs:
            log("\nПрепараты не найдены в графе:")
            for drug in missing_drugs:
                log(f"- {drug}")
        
        # Логируем отсутствующие эффекты
        if missing_effects:
            log("\nЭффекты не найдены в графе:")
            for drug, effects in missing_effects.items():
                log(f"Для '{drug}':")
                for effect in effects:
                    log(f"  - {effect}")
        
        return drug_effects
    
    def _prepare_optimization_data(self):
        """Подготовка данных для оптимизации"""
        self.param_map = []
        self.initial_params = []
        self.target_pairs = []
        
        # 1. Собираем параметры для всех узлов графа
        for node in self.graph['nodes']:
            probs = self.initial_probs.get(node['name'], {})
            for key, value in probs.items():
                self.param_map.append((node['name'], key))
                self.initial_params.append(value)
        
        # 2. Создаем целевые пары только для найденных препаратов
        total_pairs = 0
        for drug, effects in self.drug_map.items():
            for effect_id, effect_name in effects:
                target_prob = self.target_effects[drug][effect_name]
                self.target_pairs.append((effect_id, target_prob))
                total_pairs += 1
        
        log(f"\nВсего подготовлено {total_pairs} целевых пар для оптимизации")
        log(f"Всего параметров для оптимизации: {len(self.initial_params)}")

        print("param_map:", self.param_map)
    
    def optimize(self):
        """Процесс оптимизации"""
        if not self.target_pairs:
            log("\nНет данных для оптимизации!")
            return
        
        # print("target_pairs:", self.target_pairs)
        
        log("\nНачало оптимизации...")

        def create_loss_with_progress_bar(loss_fn, total_evals):
            pbar = tqdm(total=total_evals)

            def wrapped_loss(params):
                pbar.update(1)
                return loss_fn(params)

            return wrapped_loss

        wrapped_loss = create_loss_with_progress_bar(self.calculate_loss, total_evals=15000)
        
        # result = minimize(
        #     self.calculate_loss,
        #     self.initial_params,
        #     method='L-BFGS-B',
        #     bounds=self.bounds,
        #     options={'maxiter': 500}
        # )

        result = minimize(
            wrapped_loss,
            self.initial_params,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={
                'maxiter': 5000,
                'maxfun': 15000,
                'disp': True
            }
        )
        
        log("\nРезультаты оптимизации:")
        log(f"- Финальный loss: {result.fun:.6f}")
        log(f"- Итераций: {result.nit}")
        log(f"- Успешно: {result.success}")
        log(f"- Сообщение: {result.message}")
        
        self._save_results(result.x)
    
    def calculate_loss(self, params):
        """Расчет функции потерь"""
        probs = self.update_probs(params)
        network = build_network(self.graph, probs)
        calculated = calculate_probabilities(network)
        
        total_loss = 0.0
        log("\nДетали расчета:")
        
        for node_id, target in self.target_pairs:
            pred = calculated.get(node_id, 0.0)
            loss = (pred - target) ** 2
            total_loss += loss
            
            # Детализация расчета
            node = next(n for n in self.graph['nodes'] if n['id'] == node_id)
            log(f"{node['name']}: target={target:.6f}, pred={pred:.6f}, loss={loss:.6f}")
        
        log(f"\nОбщий loss: {total_loss:.6f}")
        return total_loss
    
    
    def update_probs(self, params):
        """Обновление вероятностей на основе оптимизированных параметров"""
        new_probs = defaultdict(dict)
        for i, (name, key) in enumerate(self.param_map):
            value = max(0.0, min(1.0, params[i]))
        
            # Добавляем нормализацию для комбинаций
            if ',' in key:  # Если есть комбинация родителей
                parent_states = tuple(map(int, key.split(',')))
                sum_prob = sum(
                    v for k, v in new_probs[name].items() 
                    if tuple(map(int, k.split(','))) == parent_states
                )
                value = value / (sum_prob + value) if (sum_prob + value) != 0 else 0.0
        
            new_probs[name][key] = value
        return new_probs
   
    
    def _save_results(self, params):
        """Сохранение результатов"""
        optimized_probs = self.update_probs(params)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(optimized_probs, f, indent=4, ensure_ascii=False)
        log(f"\nСохранено в {OUTPUT_FILE}")

def main():
    init_log()
    print("Начало оптимизации (детали в opt_calc.txt)...")
    
    try:
        graph = load_graph(GRAPH_FILE)
        target_effects = load_target_effects()
        
        log(f"Загружено {len(target_effects)} препаратов из {TARGET_FILE}")
        
        optimizer = ProbabilityOptimizer(graph, target_effects)
        optimizer.optimize()
        
        print(f"Оптимизация завершена. Результаты в {OUTPUT_FILE}")
        print(f"Детали расчетов в {LOG_FILE}")
    
    except Exception as e:
        log(f"ОШИБКА: {str(e)}")
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()