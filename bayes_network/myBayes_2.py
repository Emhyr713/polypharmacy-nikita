# КОММЕНТАРИЙ: Импортируем необходимые стандартные библиотеки
import json         # Для работы с файлами формата JSON
import random       # Для генерации случайных начальных вероятностей
from collections import defaultdict # Для удобного создания словарей со значениями по умолчанию
from itertools import product     # Для создания комбинаций состояний родителей
import numpy as np  # Для эффективной работы с массивами чисел (особенно в SGD)
import os           # Для работы с файловой системой (проверка существования файлов, путей)
from datetime import datetime     # Для добавления временных меток в лог-файл
from typing import Dict, List, Any, Optional
import copy


# --- Вспомогательная функция логирования ---
def log_to_file(filename, message, add_timestamp=True, init=False):
    """
    Добавляет сообщение в лог-файл.
    Помогает отслеживать, что происходит внутри программы.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if add_timestamp else ""
    prefix = f"[{timestamp_str}] " if add_timestamp else ""
    mode = 'w' if init else 'a'
    with open(filename, mode, encoding='utf-8') as f:
        f.write(f"{prefix}{message}\n")

def load_json(filename):
   """Простая функция для загрузки JSON из файла с логированием."""
   log_to_file("opt_calc.txt", f"Загрузка JSON из файла: {filename}")
   with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)
   log_to_file("opt_calc.txt", f"Успешно загружен JSON из: {filename}")
   return data


class BayesianNode:
    """
    Представляет узел (вершину) в байесовской сети.
    Каждый узел имеет свой ID, имя, список родительских узлов и
    таблицу условных вероятностей (CPT - Conditional Probability Table).
    """
    def __init__(self, node_id, name, parents, child, label, prob_data, parent_configs, cond_probs, max_parents_for_dependent=5):
        self.id = node_id
        self.name = name
        self.parents = parents
        self.child = child
        self.label = label
        self.prob_table = self._parse_probabilities(prob_data)
        self.parent_configs = parent_configs
        self.cond_probs = cond_probs
        
        # Определяем тип зависимости от родителей
        # Если количество родителей превышает порог, считаем их независимыми
        self.dependent = len(parents) <= max_parents_for_dependent if max_parents_for_dependent > 0 else True
        
        # Для отладки
        self.max_parents_for_dependent = max_parents_for_dependent

    def _parse_probabilities(self, prob_data):
        """Внутренний метод для преобразования ключей CPT из строк в кортежи чисел.
        Поддерживает входные данные как со строковыми, так и с кортежными ключами.
        """
        parsed = {}
        for k, v in prob_data.items():
            if isinstance(k, str):
                key = tuple(map(int, k.split(','))) if k else ()
            elif isinstance(k, (tuple, list)):
                key = tuple(int(x) for x in k)  # обеспечиваем целочисленность и неизменяемость
            else:
                raise TypeError(f"Неподдерживаемый тип ключа в CPT: {type(k)}")
            parsed[key] = round(float(v), 6)
        return parsed
    
    def update_prob_data_in_node(self, prob_data):
        """Обновляет таблицу вероятностей"""
        self.prob_table = self._parse_probabilities(prob_data)

    def get_probability(self, parent_states):
        """Возвращает вероятность P(Node=1) для данного состояния родителей."""
        return self.prob_table.get(parent_states, 0.0)
    

class BayesianGraph:
    def __init__(self, graph_data: dict, prob_data: Optional[dict] = None, 
                 logfile_path: str = 'opt_calc.txt', max_parents_for_dependent: int = 5):
        self.graph_data = graph_data
        self.logfile_path = logfile_path
        self.max_parents_for_dependent = max_parents_for_dependent

        # Поиск предков/потомков
        self.parent_map = defaultdict(list)
        for link in graph_data['links']:
            self.parent_map[link['target']].append(link['source'])

        self.child_map = defaultdict(list)
        for link in graph_data['links']:
            self.child_map[link['source']].append(link['target'])

        if prob_data is None:
            self.prob_data = self._create_initial_probabilities_in_memory()
        else:
            self.prob_data = prob_data

        self.network = self._build_network()
        self.sort_nodes_id_list: List[str] = self._topological_sort_pure_by_predecessors()

    def _build_network(self) -> Dict[str, BayesianNode]:
        """Собирает сеть (словарь объектов BayesianNode) из данных графа и вероятностей."""
        nodes = {}
        for node_config in self.graph_data['nodes']:
            node_id = node_config['id']
            node_name = node_config['name']
            node_label = node_config['label']
            parents = self.parent_map.get(node_id, [])
            child = self.child_map.get(node_id, [])
            node_prob_data = self.prob_data.get(node_id, {})

            # Один раз при инициализации узла
            keys = list(node_prob_data.keys())
            values = list(node_prob_data.values())

            nodes[node_id] = BayesianNode(
                node_id=node_id,
                name=node_name,
                parents=parents,
                child=child,
                label=node_label,
                prob_data=node_prob_data,
                parent_configs=np.array(keys, dtype=np.int8) if keys else np.array([], dtype=np.int8),
                cond_probs=np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64),
                max_parents_for_dependent=self.max_parents_for_dependent
            )
            
            # Логируем информацию о типе зависимости узла
            if not nodes[node_id].dependent:
                log_to_file(self.logfile_path, 
                           f"Узел '{node_name}' (id: {node_id}) помечен как НЕЗАВИСИМЫЙ от родителей "
                           f"(родителей: {len(parents)}, порог: {self.max_parents_for_dependent})")
        
        log_to_file(self.logfile_path, 
                   f"Сборка словаря объектов BayesianNode завершена. Всего узлов: {len(nodes)}")
        return nodes
    
    def update_prob_data_in_network(self, prob_data):
        """Обновляет prob_data в сети"""
        for n_id, node in self.network.items():
            new_prob_data_for_node = prob_data.get(node.id)
            node.update_prob_data_in_node(new_prob_data_for_node)
        self.prob_data = prob_data
             
    def _topological_sort_pure_by_predecessors(self) -> List[str]:
        """
        Выполняет топологическую сортировку ориентированного ациклического графа,
        заданного картой родителей. Полный список узлов берётся из self.graph_data.
        """
        all_node_ids = {node['id'] for node in self.graph_data['nodes']}

        indegree = {node: len(self.parent_map.get(node, [])) for node in all_node_ids}

        # Строим обратные связи: от родителя к детям
        child_map = defaultdict(list)
        for child, parents in self.parent_map.items():
            for parent in parents:
                child_map[parent].append(child)

        zero_indegree = [node for node in all_node_ids if indegree[node] == 0]
        result = []
        while zero_indegree:
            current = zero_indegree.pop(0)
            result.append(current)
            for child in child_map[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    zero_indegree.append(child)

        if len(result) != len(all_node_ids):
            raise ValueError("Граф содержит цикл и не может быть топологически отсортирован.")

        return result

    def _create_initial_probabilities_in_memory(self) -> dict:
        """Генерирует начальные случайные CPT для всех узлов сети на основе структуры графа."""
        log_to_file(self.logfile_path, "Генерация начальных случайных CPT в памяти.")
        probabilities = {}
        for node_config in self.graph_data['nodes']:
            node_id = node_config['id']
            parents = self.parent_map.get(node_id, [])
            
            # Определяем, является ли узел зависимым от родителей
            is_dependent = len(parents) <= self.max_parents_for_dependent if self.max_parents_for_dependent > 0 else True
            
            if not is_dependent:
                # Для узлов с независимыми родителями создаем пустую CPT
                # Вероятность будет вычисляться как произведение вероятностей родителей
                probabilities[node_id] = {}
                log_to_file(self.logfile_path, 
                           f"Узел '{node_config['name']}' (id: {node_id}) имеет {len(parents)} родителей > {self.max_parents_for_dependent}. "
                           f"CPT не генерируется, вероятность будет вычисляться как произведение вероятностей родителей.")
            elif not parents:
                # Для корневого узла ключ — пустой кортеж ()
                probabilities[node_id] = {(): round(random.uniform(0.01, 0.99), 6)}
            else:
                combs = product([0, 1], repeat=len(parents))
                # Ключи — кортежи целых чисел
                probabilities[node_id] = {
                    c: round(random.uniform(0.01, 0.99), 6) for c in combs
                }

        log_to_file(self.logfile_path, "Начальные CPT сгенерированы.")
        return probabilities
    

class BayesianNetworkOptimizer:
    """
    Основной класс, который управляет процессом оптимизации.
    """
    def __init__(self, bayes_graph, orlov_dataset, logfile_path='opt_calc.txt'):
        self.logfile_path = logfile_path

        log_to_file(self.logfile_path, "\n--- Инициализация Оптимизатора Началась ---")
        self.graph = bayes_graph
        self.orlov_data = orlov_dataset
        self.name_to_id = {node.name: n_id for n_id, node in self.graph.network.items()}

        # Запускаем проверку на соответствие данных перед началом работы
        self._pre_check_data_consistency()

        self._initialize_probabilities_and_mapping()
        log_to_file(self.logfile_path, "--- Инициализация Оптимизатора Завершена ---\n")

    def _pre_check_data_consistency(self):
        """Проверяет, что все сущности из датасета Orlov существуют в графе."""
        log_to_file(self.logfile_path, "  Запуск проверки соответствия данных графа и датасета Orlov...")
        graph_node_names = set(self.name_to_id.keys())

        found_drugs = []
        missing_drugs = []
        found_ses = set()
        missing_se_count = 0

        for drug_name, se_list in self.orlov_data.items():
            if drug_name in graph_node_names:
                found_drugs.append(drug_name)
            else:
                missing_drugs.append(drug_name)

            for se_name, _ in se_list:
                if se_name in graph_node_names:
                    found_ses.add(se_name)
                else:
                    missing_se_count += 1

        # Логирование результатов
        log_to_file(self.logfile_path, f"  Найдено препаратов: {len(found_drugs)}")
        if found_drugs:
            drugs_str = ", ".join(sorted(found_drugs))
            log_to_file(self.logfile_path, f"  Список найденных препаратов: {drugs_str}")

        log_to_file(self.logfile_path, f"  Не найдено препаратов: {len(missing_drugs)}")

        log_to_file(self.logfile_path, f"  Найдено побочных эффектов (уникальных): {len(found_ses)}")
        if found_ses:
            ses_str = ", ".join(sorted(found_ses))
            log_to_file(self.logfile_path, f"  Список найденных побочных эффектов: {ses_str}")

        log_to_file(self.logfile_path, f"  Не найдено побочных эффектов (записей в датасете): {missing_se_count}")

        # Итоговое сообщение
        if len(missing_drugs) == 0 and missing_se_count == 0:
            log_to_file(self.logfile_path, "  Проверка успешно пройдена. Все сущности из датасета Orlov найдены в графе.")
        else:
            log_to_file(self.logfile_path, "  Проверка завершена с предупреждениями. Оптимизация будет продолжена, но отсутствующие сущности будут проигнорированы.")

    def calculate_probabilities_with_evidence(self, evidence=None):
        """
        Вычисляет маргинальные вероятности P(Node=1) для всех узлов в сети,
        учитывая предоставленные "доказательства" (evidence).
        Для узлов с независимыми родителями вероятность вычисляется как произведение
        вероятностей активного состояния родителей.
        """
        if evidence is None:
            evidence = {}

        sorted_nodes = self.graph.sort_nodes_id_list
        network = self.graph.network

        probabilities = {}

        for node_id in sorted_nodes:
            node = network[node_id]

            if node_id in evidence:
                probabilities[node_id] = float(evidence[node_id])
                continue

            if not node.parents:
                # Для корневого узла вероятность задаётся конфигурацией пустого кортежа
                probabilities[node_id] = float(node.get_probability(()))
                continue

            # Проверяем, является ли узел зависимым от совместного распределения родителей
            if node.dependent:
                # Оригинальный расчет с использованием CPT
                parent_configs = node.parent_configs
                cond_probs = node.cond_probs

                # Собираем вероятности родителей в порядке, соответствующем parent_configs
                p_parents = np.array([probabilities[parent_id] for parent_id in node.parents])

                # Транслируем p_parents до формы (N, P)
                p_expanded = np.broadcast_to(p_parents, parent_configs.shape)

                # Для каждой ячейки: если состояние родителя = 1 — берём p, иначе — (1 - p)
                prob_matrix = np.where(parent_configs == 1, p_expanded, 1.0 - p_expanded)

                # Произведение по родителям даёт вероятность каждой конфигурации
                prob_parents_config = np.prod(prob_matrix, axis=1)

                # Взвешенная сумма: Σ P(Node=1 | config) * P(config)
                expected_prob = np.sum(cond_probs * prob_parents_config)
                probabilities[node_id] = round(float(expected_prob), 6)
            else:
                # Для узлов с независимыми родителями: P(Node=1) = Π P(Parent_i=1)
                prob_product = 1.0
                for parent_id in node.parents:
                    if parent_id not in probabilities:
                        error_msg = f"ОШИБКА: Не найдена вероятность для родителя '{parent_id}' узла '{node_id}'."
                        raise ValueError(error_msg)
                    prob_product *= probabilities[parent_id]
                probabilities[node_id] = round(float(prob_product), 6)

        return probabilities

    def _initialize_probabilities_and_mapping(self):
        """Готовит CPT и разделяет параметры на обучаемые и фиксированные."""
        log_to_file(self.logfile_path, "  Подготовка CPT и разделение параметров...")

        initial_prob_data = self.graph.prob_data
        id_to_node_obj = self.graph.network

        idx = 0
        self.optimizable_prob_map = {}
        self.fixed_prob_map = {}

        for node_id, cpt_dict in initial_prob_data.items():
            node = id_to_node_obj[node_id]
            
            # Для узлов с независимыми родителями не добавляем параметры в оптимизацию
            if not node.dependent:
                log_to_file(self.logfile_path, 
                           f"  Узел '{node.name}' (id: {node_id}) пропущен в оптимизации "
                           f"(независимые родители, {len(node.parents)} родителей)")
                continue
            
            is_side_e = (node.label == 'side_e')
            parents = self.graph.parent_map.get(node_id, [])
            
            for conf_str, prob in cpt_dict.items():
                conf_tuple = tuple(int(x) for x in conf_str)
                is_fixed = False
                
                if is_side_e and parents:
                    for i, p_id in enumerate(parents):
                        parent_node = id_to_node_obj.get(p_id)

                        if not parent_node or parent_node.label != 'prepare':
                            continue
                        if conf_tuple[i] != 1:
                            continue

                        drug_name = parent_node.name
                        se_name = node.name

                        # Проверка наличия препарата и совпадающего побочного эффекта в Orlov
                        if drug_name in self.orlov_data:
                            for orlov_se, orlov_prob in self.orlov_data[drug_name]:
                                if orlov_se == se_name:
                                    self.fixed_prob_map[(node_id, conf_str)] = orlov_prob
                                    is_fixed = True
                                    break

                        if is_fixed:
                            break
                
                if not is_fixed:
                    self.optimizable_prob_map[(node_id, conf_str)] = idx
                    idx += 1
                    
        self.x0 = np.array([
            initial_prob_data[k[0]][k[1]]
            for k, v in sorted(self.optimizable_prob_map.items(), key=lambda i: i[1])
        ])
        
        log_to_file(self.logfile_path, f"  Найдено {len(self.optimizable_prob_map)} обучаемых параметров.")
        log_to_file(self.logfile_path, f"  Найдено {len(self.fixed_prob_map)} фиксированных параметров из Orlov.json.")
        
        # Логируем информацию о независимых узлах
        independent_nodes = [node.name for node in self.graph.network.values() if not node.dependent]
        if independent_nodes:
            log_to_file(self.logfile_path, 
                       f"  Независимые узлы (вероятность = произведение вероятностей родителей): {', '.join(independent_nodes)}")

    def _unflatten_probabilities(self, x_flat):
        """Преобразует плоский вектор параметров обратно в полную структуру CPT с округлением."""
        current_prob_data = copy.deepcopy(self.graph.prob_data)
        
        # Обновление оптимизируемых параметров только для зависимых узлов
        for (node_id, conf_key), idx in self.optimizable_prob_map.items():
            current_prob_data[node_id][conf_key] = round(float(x_flat[idx]), 6)
        
        # Восстановление фиксированных параметров (если они были заменены)
        for (node_id, conf_key), val in self.fixed_prob_map.items():
            current_prob_data[node_id][conf_key] = round(float(val), 6)
        
        return current_prob_data
    
    def _calculate_loss(self, x_flat, log_details=False):
        """
        Функция потерь (loss function).
        Если log_details=True, выводит в лог сравнение текущих и целевых значений.
        """
        current_prob_data = self._unflatten_probabilities(x_flat)
        self.graph.update_prob_data_in_network(current_prob_data)

        total_loss = 0.0
        prepare_nodes = {node.name: n_id for n_id, node in self.graph.network.items() if node.label == 'prepare'}
        side_e_nodes = {node.name: n_id for n_id, node in self.graph.network.items() if node.label == 'side_e'}

        if log_details:
            log_to_file(self.logfile_path, "    --- Детальный расчет ошибки ---", add_timestamp=False)

        for drug_name, se_data_list in self.orlov_data.items():
            drug_id = prepare_nodes.get(drug_name)
            if not drug_id: 
                continue
            
            evidence = {drug_id: 1.0}
            calculated_marginals = self.calculate_probabilities_with_evidence(evidence)
            
            for se_name, target_prob in se_data_list:
                se_id = side_e_nodes.get(se_name)
                if not se_id: 
                    continue
                
                calculated_prob = calculated_marginals.get(se_id, 0.0)
                loss_component = (calculated_prob - target_prob)**2
                total_loss += loss_component
                
                if log_details:
                    log_msg = (f"      - Препарат: '{drug_name}', Побочный эффект: '{se_name}'\n"
                               f"        -> Текущее значение: {calculated_prob:.6f} | "
                               f"Целевое значение: {target_prob:.6f} | "
                               f"Квадрат ошибки: {loss_component:.6f}")
                    log_to_file(self.logfile_path, log_msg, add_timestamp=False)
        
        if log_details:
            log_to_file(self.logfile_path, "    ---------------------------------", add_timestamp=False)

        return total_loss
    
    def _calculate_gradients(self, x_flat):
        """Вычисляет градиент функции потерь по каждому параметру."""
        grads = np.zeros_like(x_flat)
        h = 1e-6
        for i in range(len(x_flat)):
            x_plus_h, x_minus_h = x_flat.copy(), x_flat.copy()
            x_plus_h[i] += h
            x_minus_h[i] -= h
            loss_plus_h = self._calculate_loss(x_plus_h)
            loss_minus_h = self._calculate_loss(x_minus_h)
            grads[i] = (loss_plus_h - loss_minus_h) / (2 * h)
        
        return grads

    def optimize_sgd(self, learning_rate=0.01, epochs=500, log_interval=10, intermediate_file='intermediate_probabilities_opt.json'):
        """Запускает процесс оптимизации с помощью стохастического градиентного спуска (SGD)."""
        log_to_file(self.logfile_path, "\n--- Запуск Оптимизации (метод SGD) ---")
        log_to_file(self.logfile_path, f"  Параметры: Скорость обучения={learning_rate}, Эпохи={epochs}")
        log_to_file(self.logfile_path, f"  Порог для независимых узлов: {self.graph.max_parents_for_dependent}")
        
        current_params = self.x0.copy()

        log_to_file(self.logfile_path, "  --- Начальные значения оптимизируемых переменных ---", add_timestamp=False)
        sorted_params = sorted(self.optimizable_prob_map.items(), key=lambda item: item[1])

        for (node_id, conf_str), idx in sorted_params:
            node_name = self.graph.network[node_id].name
            log_to_file(self.logfile_path, 
                       f"    P({node_name}|{conf_str if conf_str else '""'}) [idx:{idx}] = {current_params[idx]:.6f}", 
                       add_timestamp=False)
        log_to_file(self.logfile_path, "  -------------------------------------------------", add_timestamp=False)

        initial_loss = self._calculate_loss(current_params)
        log_to_file(self.logfile_path, f"  Начальная ошибка (Loss): {initial_loss:.8f}\n")

        for epoch in range(epochs):
            gradients = self._calculate_gradients(current_params)
            current_params -= learning_rate * gradients
            np.clip(current_params, 0.01, 0.99, out=current_params)

            if (epoch + 1) % log_interval == 0:
                log_to_file(self.logfile_path, f"  Эпоха {epoch + 1}/{epochs}:")
                current_loss = self._calculate_loss(current_params, log_details=True)
                log_to_file(self.logfile_path, f"    Общая ошибка (Loss) = {current_loss:.8f}\n", add_timestamp=False)

                # Чекпоинт
                intermediate_prob_data = self._unflatten_probabilities(current_params)
                save_optimized_probabilities(intermediate_prob_data, intermediate_file)

        log_to_file(self.logfile_path, "--- Оптимизация (метод SGD) Завершена ---")
        final_loss = self._calculate_loss(current_params)
        log_to_file(self.logfile_path, f"  Итоговая ошибка (Loss): {final_loss:.8f}")
        final_prob_data = self._unflatten_probabilities(current_params)
        return final_prob_data, final_loss


def save_optimized_probabilities(optimized_prob_data, filename="probabilities_opt.json"):
    """Сохраняет все оптимизированные таблицы CPT в отдельный файл."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(optimized_prob_data, f, ensure_ascii=False, indent=4)


def debag_save_optimized_probabilities(graph, filename="debug_probabilities_opt.json"):
    """Сохраняет все оптимизированные таблицы CPT в отдельный файл с метаданными узла"""
    network = graph.network
    sorted_nodes = graph.sort_nodes_id_list

    result = {}

    for node_id in sorted_nodes:
        node = network[node_id]
        
        result[node_id] = {
            "name": node.name,
            "parents_ids": node.parents,
            "parents_names": [network[par_id].name for par_id in node.parents],
            "dependent": node.dependent,
            "num_parents": len(node.parents),
            "max_parents_for_dependent": node.max_parents_for_dependent,
            "prob_table": {",".join(map(str, k)): v for k, v in node.prob_table.items()}
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    DIR = "bayes_network\\"

    graph_path      = f'{DIR}data\\graphs\\graphs_4.json'
    orlov_path      = f'{DIR}data\\Orlov.json'
    logfile_path    = f'{DIR}data\\opt_calc_mybayes_2_graphs_4.txt'
    
    # Параметр, определяющий порог для независимых узлов
    # Если количество родителей > max_parents_for_dependent, узел считается независимым
    max_parents_for_dependent = 5  # Можно изменить это значение
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_from_json = json.load(f)

    with open(orlov_path, 'r', encoding='utf-8') as f:
        orlov_dataset = json.load(f)

    log_to_file(logfile_path, "", init=True)
    
    # Создаем сеть с указанием порога для независимых узлов
    network = BayesianGraph(
        graph_from_json, 
        logfile_path=logfile_path,
        max_parents_for_dependent=max_parents_for_dependent
    )
    
    optimazer = BayesianNetworkOptimizer(network, orlov_dataset, logfile_path=logfile_path)

    final_prob_data, final_loss = optimazer.optimize_sgd(epochs=10, log_interval=2)

    save_optimized_probabilities(final_prob_data, f'{DIR}data\\probabilities_opt.json')
    debag_save_optimized_probabilities(network, f'{DIR}data\\debug_probabilities_opt.json')