# КОММЕНТАРИЙ: Импортируем необходимые стандартные библиотеки
import json         # Для работы с файлами формата JSON
import random       # Для генерации случайных начальных вероятностей
from collections import defaultdict # Для удобного создания словарей со значениями по умолчанию
from itertools import product     # Для создания комбинаций состояний родителей
import numpy as np  # Для эффективной работы с массивами чисел (особенно в SGD)
import os           # Для работы с файловой системой (проверка существования файлов, путей)
from datetime import datetime     # Для добавления временных меток в лог-файл


# --- Вспомогательная функция логирования ---
def log_to_file(filename, message, add_timestamp=True):
    """
    Добавляет сообщение в лог-файл.
    Помогает отслеживать, что происходит внутри программы.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if add_timestamp else ""
    prefix = f"[{timestamp_str}] " if add_timestamp else ""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"{prefix}{message}\n")

# --- Основные классы и функции байесовской сети ---

class BayesianNode:
    """
    Представляет узел (вершину) в байесовской сети.
    Каждый узел имеет свой ID, имя, список родительских узлов и
    таблицу условных вероятностей (CPT - Conditional Probability Table).
    """
    def __init__(self, node_id, name, parents, prob_data):
        self.id = node_id
        self.name = name
        self.parents = parents
        self.prob_table = self._parse_probabilities(prob_data)

    def _parse_probabilities(self, prob_data):
        """Внутренний метод для преобразования ключей CPT из строк в кортежи чисел."""
        parsed = {}
        for k, v in prob_data.items():
            key = tuple(map(int, k.split(','))) if k else ()
            parsed[key] = float(v)
        return parsed

    def get_probability(self, parent_states):
        """Возвращает вероятность P(Node=1) для данного состояния родителей."""
        return self.prob_table.get(parent_states, 0.0)

def topological_sort(nodes_to_sort, parent_map):
    """
    Выполняет топологическую сортировку узлов.
    Это нужно, чтобы вычислять вероятности в правильном порядке: от "предков" к "потомкам".
    """
    print("nodes_to_sort:", nodes_to_sort, "\n\n")
    print("parent_map:", parent_map, "\n\n")
    in_degree = {nid: 0 for nid in nodes_to_sort}
    children_map = defaultdict(list)
    for child, parents in parent_map.items():
        if child in nodes_to_sort:
            for parent in parents:
                if parent in nodes_to_sort:
                    in_degree[child] += 1
                    children_map[parent].append(child)

    
    queue = [nid for nid in nodes_to_sort if in_degree[nid] == 0]
    print("children_map", children_map, "\n\n")
    print("queue", queue, "\n\n")
    sorted_list = []
    while queue:
        u = queue.pop(0)
        sorted_list.append(u)
        for v in children_map[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(sorted_list) != len(nodes_to_sort):
        log_to_file("opt_calc.txt", "ОШИБКА: Граф содержит циклы, что недопустимо для байесовской сети.")
        raise ValueError("Граф содержит циклы, что недопустимо для байесовской сети.")
    

    print("sorted_list:", sorted_list)

    return sorted_list


# def show_graph(data):

    
#     # Выбор алгоритма распределения узлов
#     pos = nx.shell_layout(G)

#     # Визуализация графа
#     plt.figure(figsize=(10, 8))
#     nx.draw(
#         G,
#         pos,
#         with_labels=True,
#         node_color='lightblue',
#         node_size=500,
#         font_size=10,
#         font_weight='bold',
#         edge_color='gray'
#     )
#     plt.title("Визуализация графа")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# Проверено
def calculate_probabilities_with_evidence(network_nodes, evidence=None, topological_sorted_list = None):
    """
    Вычисляет маргинальные вероятности P(Node=1) для всех узлов в сети,
    учитывая предоставленные "доказательства" (evidence).
    """
    if evidence is None: evidence = {}
    network = {node_id: node for node_id, node in network_nodes.items()}
    # parent_map = {nid: node.parents for nid, node in network.items()}
    # sorted_nodes = topological_sort(list(network.keys()), parent_map)


    sorted_nodes = topological_sorted_list

    # print("network.items():", network.items(), "\n\n")
    # print("sorted_nodes:", sorted_nodes, "\n\n")

    probabilities = {}
    for node_id in sorted_nodes:
        node = network[node_id]
        if node_id in evidence:
            probabilities[node_id] = evidence[node_id]
            continue
        if not node.parents:
            probabilities[node_id] = node.get_probability(())
            continue
        expected_prob_sum = 0.0

        # print("node.prob_table.items", node.prob_table.items())  
        
        # print("node:", node_id, node.name)  
        # print("parents:", node.parents, [network[parent_id].name for parent_id in node.parents])  
        for parent_states_tuple, prob_node_given_parents in node.prob_table.items():

            
            # print("prob_node_given_parents", parent_states_tuple, prob_node_given_parents)

            prob_parents_config = 1.0
            # print("prob_parents_config:", prob_parents_config)
            for i, parent_id in enumerate(node.parents):
                parent_state_value = parent_states_tuple[i]
                if parent_id not in probabilities:
                    error_msg = f"ОШИБКА: Не найдена вероятность для родителя '{parent_id}' узла '{node_id}'."
                    log_to_file("opt_calc.txt", error_msg)
                    raise ValueError(error_msg)
                prob_parent_is_1 = probabilities[parent_id]

                # print("\t", parent_state_value, parent_id, network[parent_id].name, "p=", end='')


                if parent_state_value == 1:
                    prob_parents_config *= prob_parent_is_1
                    # print("p = ", prob_parent_is_1)
                else:
                    prob_parents_config *= (1 - prob_parent_is_1)
                    # print(f"1-p = (1 - {prob_parent_is_1}) = {1 - prob_parent_is_1}")

                # print("prob_parents_config:", prob_parents_config)
            
            # print("expected_prob_cur", prob_node_given_parents * prob_parents_config)
            expected_prob_sum += prob_node_given_parents * prob_parents_config
            # print("---")
            # print("expected_prob_sum_temp:", expected_prob_sum)

            # print("\n")
# 
        # print("-------------------------------------------")
        # print(f"{node_id}, {node.name} expected_prob_sum:", expected_prob_sum)
# 
        # print("\n\n")
        probabilities[node_id] = expected_prob_sum


    return probabilities

# --- Функции для работы с файлами ---

def load_json(filename):
   """Простая функция для загрузки JSON из файла с логированием."""
   log_to_file("opt_calc.txt", f"Загрузка JSON из файла: {filename}")
   with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)
   log_to_file("opt_calc.txt", f"Успешно загружен JSON из: {filename}")
   return data


# Проверено
def create_initial_probabilities_in_memory(graph_data):
   """Генерирует начальные случайные CPT для всех узлов сети."""
   log_to_file("opt_calc.txt", "Генерация начальных случайных CPT в памяти.")
   parent_map = defaultdict(list)
   for link in graph_data['links']:
    parent_map[link['target']].append(link['source'])
   probabilities = {}
   for node_config in graph_data['nodes']:
    node_id, parents, node_name = node_config['id'], parent_map.get(node_config['id'], []), node_config['name']
    if not parents:
     probabilities[node_name] = {"": random.uniform(0.01, 0.99)}
    else:
     combs = product([0, 1], repeat=len(parents))
     probabilities[node_name] = {','.join(map(str, c)): random.uniform(0.01, 0.99) for c in combs}

   print(probabilities)
   log_to_file("opt_calc.txt", "Начальные CPT сгенерированы.")
   return probabilities

def build_network(graph_data, prob_data):
   """Собирает сеть (словарь объектов BayesianNode) из данных графа и вероятностей."""
   parent_map = defaultdict(list)
   for link in graph_data['links']:
    parent_map[link['target']].append(link['source'])
   nodes = {}
   for node_config in graph_data['nodes']:
    node_id, node_name = node_config['id'], node_config['name']
    nodes[node_id] = BayesianNode(
     node_id=node_id, name=node_name, parents=parent_map.get(node_id, []),
     prob_data=prob_data.get(node_name, {})
    )
   return nodes

def load_drug_states(drug_states_path):
    """Загружает состояния препаратов из файла."""
    log_to_file("opt_calc.txt", f"Загрузка состояний препаратов из файла: {drug_states_path}")
    with open(drug_states_path, 'r', encoding='utf-8') as f:
        drug_states = json.load(f)
    log_to_file("opt_calc.txt", f"Успешно загружены состояния препаратов из: {drug_states_path}")
    return drug_states

def save_optimized_results_with_drug_states(optimized_prob_data, graph_data, drug_states, filename="optimized_results.json"):
    """Сохраняет итоговый результат - вероятности побочных эффектов для заданной комбинации препаратов."""
    log_to_file("opt_calc.txt", f"Сохранение результатов оптимизации в: {filename}")
    id_to_node_obj = {n['id']: n for n in graph_data['nodes']}
    optimized_network_nodes = build_network(graph_data, optimized_prob_data)
    side_e_nodes_info = {n['id']: n['name'] for n in graph_data['nodes'] if n.get('label') == 'side_e'}
    evidence = {node_id: state for node_id, state in drug_states.items()}
    drug_names = [id_to_node_obj[drug_id]['name'] for drug_id in drug_states.keys()]
    descriptive_combination_name = " + ".join([f"{name}={drug_states[id]}" for name, id in zip(drug_names, drug_states.keys())])
    result_entry = {
        "combination_description": descriptive_combination_name,
        "drugs": drug_names,
        "drug_states_input": drug_states,
        "side_effects": {}
    }
    calculated_marginals = calculate_probabilities_with_evidence(optimized_network_nodes, evidence)
    for se_id, se_name in side_e_nodes_info.items():
        result_entry["side_effects"][se_name] = {
            "id": se_id,
            "probability": round(calculated_marginals.get(se_id, 0.0), 6)
        }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_entry, f, ensure_ascii=False, indent=4)
    log_to_file("opt_calc.txt", f"Результаты успешно сохранены в: {filename}")

def save_optimized_probabilities(optimized_prob_data, filename="probabilities_opt.json"):
    """Сохраняет все оптимизированные таблицы CPT в отдельный файл."""
    log_to_file("opt_calc.txt", f"Сохранение оптимизированных CPT в: {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(optimized_prob_data, f, ensure_ascii=False, indent=4)
    log_to_file("opt_calc.txt", f"Оптимизированные CPT сохранены в: {filename}")

def save_graph_with_optimized_weights(original_graph_data, optimized_prob_data, graph_path):
    """Сохраняет новый файл графа, где 'weight' каждого узла равен его базовой вероятности P(Node=1)."""
    base_file_name = os.path.splitext(os.path.basename(graph_path))[0]
    filename = f"{base_file_name}_optimized_weights.json"
    log_to_file("opt_calc.txt", f"Сохранение графа с обновленными весами в: {filename}")
    optimized_network_nodes = build_network(original_graph_data, optimized_prob_data)
    all_node_marginal_probs = calculate_probabilities_with_evidence(optimized_network_nodes, evidence=None)
    updated_graph_data = json.loads(json.dumps(original_graph_data))
    for node_config in updated_graph_data['nodes']:
        node_id = node_config['id']
        if node_id in all_node_marginal_probs:
            node_config['weight'] = round(all_node_marginal_probs[node_id], 3)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(updated_graph_data, f, ensure_ascii=False, indent=4)
    log_to_file("opt_calc.txt", f"Граф с весами успешно сохранен в: {filename}")
    return filename

# --- Класс Оптимизатора Байесовской Сети ---

class BayesianNetworkOptimizer:
    """
    Основной класс, который управляет процессом оптимизации.
    """
    def __init__(self, graph_path, orlov_data_path, initial_prob_path=None):
        log_to_file("opt_calc.txt", "\n--- Инициализация Оптимизатора ---")
        self.graph_data = load_json(graph_path)
        self.orlov_data = load_json(orlov_data_path)
        self.id_to_node_obj = {n['id']: n for n in self.graph_data['nodes']}
        self.name_to_id = {n['name']: n['id'] for n in self.graph_data['nodes']}
        self.parent_map = defaultdict(list)

        for link in self.graph_data['links']:
            self.parent_map[link['target']].append(link['source'])

        # Отсортировать сразу
        self.sorted_nodes = topological_sort([node['id'] for node in self.graph_data], self.parent_map)
        
        # Запускаем проверку на соответствие данных перед началом работы
        self._pre_check_data_consistency()
        
        self._initialize_probabilities_and_mapping(initial_prob_path)
        log_to_file("opt_calc.txt", "--- Инициализация Оптимизатора Завершена ---\n")

    def _pre_check_data_consistency(self):
        """Проверяет, что все сущности из датасета Orlov существуют в графе."""
        log_to_file("opt_calc.txt", "  Запуск проверки соответствия данных графа и датасета Orlov...")
        graph_node_names = {node['name'] for node in self.graph_data['nodes']}
        found_issue = False

        for drug_name, se_list in self.orlov_data.items():
            if drug_name not in graph_node_names:
                log_to_file("opt_calc.txt", f"  ПРЕДУПРЕЖДЕНИЕ: Препарат '{drug_name}' из датасета Orlov не найден в графе.")
                found_issue = True
            for se_name, _ in se_list:
                if se_name not in graph_node_names:
                    log_to_file("opt_calc.txt", f"  ПРЕДУПРЕЖДЕНИЕ: Побочный эффект '{se_name}' (для препарата '{drug_name}') из датасета Orlov не найден в графе.")
                    found_issue = True
        
        if not found_issue:
            log_to_file("opt_calc.txt", "  Проверка успешно пройдена. Все сущности из датасета Orlov найдены в графе.")
        else:
             log_to_file("opt_calc.txt", "  Проверка завершена с предупреждениями. Оптимизация будет продолжена, но отсутствующие сущности будут проигнорированы.")

    def _initialize_probabilities_and_mapping(self, initial_prob_path):
        """Готовит CPT и разделяет параметры на обучаемые и фиксированные."""
        log_to_file("opt_calc.txt", "  Подготовка CPT и разделение параметров...")
        if initial_prob_path and os.path.exists(initial_prob_path):
            self.initial_prob_data = load_json(initial_prob_path)
        else:
            self.initial_prob_data = create_initial_probabilities_in_memory(self.graph_data)

        print("self.graph_data:", self.graph_data, "\n\n")
        print("self.initial_prob_data:", self.initial_prob_data, "\n\n")

        
        idx = 0
        self.optimizable_prob_map = {}
        self.fixed_prob_map = {}
        
        for node_name, cpt_dict in self.initial_prob_data.items():
            node_id = self.name_to_id.get(node_name)
            if not node_id: continue
            
            is_side_e = (self.id_to_node_obj[node_id].get('label') == 'side_e')
            parents = self.parent_map.get(node_id, [])
            
            for conf_str, prob in cpt_dict.items():
                is_fixed = False
                conf_tuple = tuple(map(int, conf_str.split(','))) if conf_str else ()
                
                if is_side_e and parents:
                    for i, p_id in enumerate(parents):
                        p_node = self.id_to_node_obj.get(p_id)
                        if p_node and p_node.get('label') == 'prepare' and conf_tuple[i] == 1:
                            drug_name, se_name = p_node['name'], node_name
                            if drug_name in self.orlov_data:
                                for orlov_se, orlov_prob in self.orlov_data[drug_name]:
                                    if orlov_se == se_name:
                                        self.fixed_prob_map[(node_name, conf_str)] = orlov_prob
                                        is_fixed = True
                                        break
                            if is_fixed: break
                
                if not is_fixed:
                    self.optimizable_prob_map[(node_name, conf_str)] = idx
                    idx += 1
                    
        self.x0 = np.array([
            self.initial_prob_data[k[0]][k[1]]
            for k, v in sorted(self.optimizable_prob_map.items(), key=lambda i: i[1])
        ])
        
        log_to_file("opt_calc.txt", f"  Найдено {len(self.optimizable_prob_map)} обучаемых параметров.")
        log_to_file("opt_calc.txt", f"  Найдено {len(self.fixed_prob_map)} фиксированных параметров из Orlov.json.")

    def _unflatten_probabilities(self, x_flat):
        """Преобразует плоский вектор параметров обратно в полную структуру CPT."""
        current_prob_data = json.loads(json.dumps(self.initial_prob_data))
        for (node_name, conf_str), idx in self.optimizable_prob_map.items():
            current_prob_data[node_name][conf_str] = x_flat[idx]
        for (node_name, conf_str), val in self.fixed_prob_map.items():
            current_prob_data[node_name][conf_str] = val

        print("current_prob_data:", current_prob_data, end="\n\n\n\n\n")
        return current_prob_data

    def _calculate_loss(self, x_flat, log_details=False):
        """
        Функция потерь (loss function).
        Если log_details=True, выводит в лог сравнение текущих и целевых значений.
        """
        current_prob_data = self._unflatten_probabilities(x_flat)
        network = build_network(self.graph_data, current_prob_data)

        total_loss = 0.0
        prepare_nodes = {n['name']: n['id'] for n in self.graph_data['nodes'] if n.get('label') == 'prepare'}
        side_e_nodes = {n['name']: n['id'] for n in self.graph_data['nodes'] if n.get('label') == 'side_e'}

        if log_details:
             log_to_file("opt_calc.txt", "    --- Детальный расчет ошибки ---", add_timestamp=False)

        for drug_name, se_data_list in self.orlov_data.items():
            drug_id = prepare_nodes.get(drug_name)
            if not drug_id: continue
            
            evidence = {drug_id: 1.0}
            calculated_marginals = calculate_probabilities_with_evidence(network, evidence, self.sorted_nodes)
            
            for se_name, target_prob in se_data_list:
                se_id = side_e_nodes.get(se_name)
                if not se_id: continue
                
                calculated_prob = calculated_marginals.get(se_id, 0.0)
                loss_component = (calculated_prob - target_prob)**2
                total_loss += loss_component
                
                if log_details:
                    log_msg = (f"      - Препарат: '{drug_name}', Побочный эффект: '{se_name}'\n"
                               f"        -> Текущее значение: {calculated_prob:.6f} | "
                               f"Целевое значение: {target_prob:.6f} | "
                               f"Квадрат ошибки: {loss_component:.6f}")
                    log_to_file("opt_calc.txt", log_msg, add_timestamp=False)
        
        if log_details:
             log_to_file("opt_calc.txt", "    ---------------------------------", add_timestamp=False)

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

    def optimize_sgd(self, learning_rate=0.01, epochs=500, log_interval=10):
        """Запускает процесс оптимизации с помощью стохастического градиентного спуска (SGD)."""
        log_to_file("opt_calc.txt", "\n--- Запуск Оптимизации (метод SGD) ---")
        log_to_file("opt_calc.txt", f"  Параметры: Скорость обучения={learning_rate}, Эпохи={epochs}")
        
        current_params = self.x0.copy()
        
        log_to_file("opt_calc.txt", "  --- Начальные значения оптимизируемых переменных ---", add_timestamp=False)
        sorted_params = sorted(self.optimizable_prob_map.items(), key=lambda item: item[1])
        for (node_name, conf_str), idx in sorted_params:
            log_to_file("opt_calc.txt", f"    P({node_name}|{conf_str if conf_str else '""'}) [idx:{idx}] = {current_params[idx]:.6f}", add_timestamp=False)
        log_to_file("opt_calc.txt", "  -------------------------------------------------", add_timestamp=False)

        initial_loss = self._calculate_loss(current_params)
        log_to_file("opt_calc.txt", f"  Начальная ошибка (Loss): {initial_loss:.8f}\n")

        for epoch in range(epochs):
            gradients = self._calculate_gradients(current_params)
            current_params -= learning_rate * gradients
            np.clip(current_params, 0.01, 0.99, out=current_params)

            if (epoch + 1) % log_interval == 0:
                log_to_file("opt_calc.txt", f"  Эпоха {epoch + 1}/{epochs}:")
                current_loss = self._calculate_loss(current_params, log_details=True)
                log_to_file("opt_calc.txt", f"    Общая ошибка (Loss) = {current_loss:.8f}\n", add_timestamp=False)

        log_to_file("opt_calc.txt", "--- Оптимизация (метод SGD) Завершена ---")
        final_loss = self._calculate_loss(current_params)
        log_to_file("opt_calc.txt", f"  Итоговая ошибка (Loss): {final_loss:.8f}")
        final_prob_data = self._unflatten_probabilities(current_params)
        return final_prob_data, final_loss

# --- Основной блок выполнения программы ---
if __name__ == "__main__":
    # graph_path = 'bayes_network\\data\\graphs\\TEST.json'
    graph_path = 'bayes_network\\data\\graphs\\graphs_4.json'
    orlov_path = 'bayes_network\\data\\Orlov.json'
    drug_states_path = 'bayes_network\\data\\drug_states.json'
    initial_probabilities_path = 'bayes_network\\data\\pr.json'
    optimized_results_path = 'bayes_network\\data\\optimized_results.json'
    probabilities_opt_path = 'bayes_network\\data\\probabilities_opt.json'
    log_file_path = 'bayes_network\\data\\opt_calc.txt'

    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    log_to_file(log_file_path, "--- Начало новой сессии оптимизации ---")

    try:
        optimizer = BayesianNetworkOptimizer(
            graph_path=graph_path,
            orlov_data_path=orlov_path,
            initial_prob_path=initial_probabilities_path
        )

        optimized_cpts, final_loss = optimizer.optimize_sgd(
            epochs=100,
            learning_rate=0.05,
            log_interval=10 # Выводить детальный лог каждые 10 эпох
        )

        if not os.path.exists(drug_states_path):
            raise FileNotFoundError(f"Файл '{drug_states_path}' не найден. Он нужен для финального расчета.")
        drug_states_from_file = load_drug_states(drug_states_path)
        
        save_optimized_results_with_drug_states(optimized_cpts, optimizer.graph_data, drug_states_from_file, filename=optimized_results_path)
        save_optimized_probabilities(optimized_cpts, filename=probabilities_opt_path)
        graph_with_weights_filename = save_graph_with_optimized_weights(optimizer.graph_data, optimized_cpts, graph_path)
        
        log_to_file(log_file_path, f"\nВсе расчеты завершены. Итоговая ошибка: {final_loss:.8f}")
        print("\nРасчеты успешно завершены.")
        print(f"  1. Вероятности побочных эффектов для комбинации сохранены в: {optimized_results_path}")
        print(f"  2. Оптимизированные CPT сохранены в: {probabilities_opt_path}")
        print(f"  3. Граф с обновленными весами узлов сохранен в: {graph_with_weights_filename}")
        print(f"  Подробный лог выполнения доступен в файле: {log_file_path}")



    except FileNotFoundError as e:
        error_msg = f"ОШИБКА: Файл не найден - {e}"
        print(f"\n{error_msg}")
        log_to_file(log_file_path, f"КРИТИЧЕСКАЯ ОШИБКА: {error_msg}")
    except Exception as e:
        error_msg = f"Критическая ошибка при выполнении: {e}"
        print(f"\n{error_msg}")
        log_to_file(log_file_path, f"КРИТИЧЕСКАЯ ОШИБКА: {error_msg}")