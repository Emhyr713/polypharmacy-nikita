import json
import random
from collections import defaultdict
from itertools import product

class BayesianNode:
    def __init__(self, node_id, name, parents, prob_data):
        self.id = node_id
        self.name = name
        self.parents = parents
        self.prob_table = self._parse_probabilities(prob_data)

    def _parse_probabilities(self, prob_data):
        parsed = {}
        for k, v in prob_data.items():
            key = tuple(map(int, k.split(','))) if k else ()
            parsed[key] = float(v)
        return parsed

    def get_probability(self, parent_states):
        return self.prob_table.get(parent_states, 0.0)

def load_graph(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_probabilities(filename='probabilities.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    
# Поиск узлов - препаратов (TODO: В ПРЕДКАХ ОБЯЗАТЕЛЬНО GROUP)
def get_prepare_node(graph, key = 'id'):
    return [n[key] for n in graph['nodes'] if n.get('label') == 'prepare']

def create_base_probabilities(graph_data, output_file='probabilities.json'):
    # Инициализация структур данных
    parent_map = defaultdict(list)
    id_to_node = {n['id']: n for n in graph_data['nodes']}
    
    # Построение карты родителей
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])
    
    # Находим все prepare-узлы и их предков 
    prepare_nodes = get_prepare_node(graph_data, key='id')
    zero_nodes = set()
    
    # Рекурсивный поиск предков
    def find_ancestors(node_id):
        ancestors = set()
        for parent in parent_map.get(node_id, []):
            ancestors.add(parent)
            ancestors.update(find_ancestors(parent))
        return ancestors
    
    # Собираем все узлы для обнуления
    for node_id in prepare_nodes:
        zero_nodes.add(node_id)
        zero_nodes.update(find_ancestors(node_id))
    
    # Генерация вероятностей
    probabilities = {}
    for node in graph_data['nodes']:
        node_id = node['id']
        parents = parent_map.get(node_id, [])
        
        # Для препаратов и их потомков
        if node_id in zero_nodes:
            if not parents:
                probabilities[node['name']] = {"": 0.0}
            else:
                combs = product([0], repeat=len(parents))  # Все комбинации нулей
                probabilities[node['name']] = {','.join(map(str, c)): 0.0 for c in combs}
        else:
            # Случайные вероятности для остальных
            if not parents:
                probabilities[node['name']] = {"": random.uniform(0.05, 0.9)}
            else:
                combs = product([0, 1], repeat=len(parents))
                probabilities[node['name']] = {','.join(map(str, c)): random.uniform(0.05, 0.9) for c in combs}
    
    # Сохранение в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(probabilities, f, indent=4, ensure_ascii=False)

    return probabilities

# Функция создания файла доз
def create_doses_file(graph_data, output_file='prepare_doses.json'):
    prepare_nodes = get_prepare_node(graph_data, key='name')
    doses = {name: 1 for name in prepare_nodes}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(doses, f, indent=4, ensure_ascii=False)


#  Функция загрузки и объединения данных
def load_combined_data(prob_file='probabilities_opt.json', doses_file='prepare_doses.json'):
    with open(prob_file, 'r', encoding='utf-8') as f:
        probabilities = json.load(f)
    
    try:
        with open(doses_file, 'r', encoding='utf-8') as f:
            doses = json.load(f)
    except FileNotFoundError:
        return probabilities
    
    # Обновляем вероятности из файла доз
    for name, value in doses.items():
        if name in probabilities:
            # Для узлов без родителей
            if "" in probabilities[name]:
                probabilities[name][""] = float(value)
            else:
                # Для узлов с родителями (обновляем все комбинации)
                probabilities[name] = {k: float(value) for k in probabilities[name]}
    
    return probabilities


def topological_sort(nodes, parent_map):
    visited = set()
    result = []
    stack = []
    
    for node in nodes:
        if node not in visited:
            stack.append((node, False))
            while stack:
                current, processed = stack.pop()
                if processed:
                    result.append(current)
                    continue
                if current in visited:
                    continue
                visited.add(current)
                stack.append((current, True))
                for parent in reversed(parent_map.get(current, [])):
                    if parent not in visited:
                        stack.append((parent, False))
    return result

def build_network(graph_data, prob_data):
    parent_map = defaultdict(list)
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])
    
    nodes = {}
    for node in graph_data['nodes']:
        node_id = node['id']
        nodes[node_id] = BayesianNode(
            node_id=node_id,
            name=node['name'],
            parents=parent_map.get(node_id, []),
            prob_data=prob_data.get(node['name'], {})
        )
    return nodes

def calculate_probabilities(network, trace_file = 'calculation_trace.txt'):
    """Вычисляет априорные вероятности для всех узлов сети"""
    
    # 1. Создаем словарь связей "узел -> его родители"
    parent_map = {nid: node.parents for nid, node in network.items()}
    
    # 2. Топологическая сортировка узлов
    sorted_nodes = topological_sort(network.keys(), parent_map)
    
    # 3. Словарь для накопления результатов
    probabilities = {}

    # Открываем файл для записи логов
    with open(trace_file, 'w', encoding='utf-8') as f:
        # 4. Заголовок лога
        f.write("\n" + "="*50 + "\n")
        f.write("НАЧАЛО РАСЧЕТА ВЕРОЯТНОСТЕЙ\n")
        f.write(f"Порядок обработки узлов: {sorted_nodes}\n")
        f.write("="*50 + "\n\n")

        # 5. Обработка узлов
        for node_id in sorted_nodes:
            node = network[node_id]
            f.write(f"[Узел {node_id} '{node.name}']\n")
            f.write(f"Тип: {'Корневой' if not node.parents else 'Дочерний'}\n")
            f.write(f"Родители: {node.parents or 'нет'}\n")

            if not node.parents:
                # Узел без родителей
                prob = node.get_probability(())
                probabilities[node_id] = prob
                f.write(f"Вероятность из таблицы: {prob:.4f}\n")
                f.write("-"*50 + "\n\n")
                continue

            total_conditional = sum(node.prob_table.values())
            normalized_probs = {}
            
            if abs(total_conditional - 1.0) > 1e-9:
                f.write(f"! Нормализация условных вероятностей (исходная сумма: {total_conditional:.4f})\n")
                for comb, p in node.prob_table.items():
                    normalized_probs[comb] = p / total_conditional if total_conditional != 0 else 0.0
            else:
                normalized_probs = node.prob_table


            # Расчет для узлов с родителями
            total = 0.0
            # f.write(f"Комбинации состояний родителей ({len(node.prob_table)}):\n")

            f.write(f"Комбинации состояний родителей ({len(normalized_probs)}):\n")
            
            for i, (comb, p_node) in enumerate(normalized_probs.items(), 1):
                prob_comb = 1.0
                comb_str = ",".join(map(str, comb))
                f.write(f"\nКомбинация {i}: {comb_str}\n")
                f.write(f"P({node.name}|{comb_str}) = {p_node:.4f}\n")

                for j, (parent_id, state) in enumerate(zip(node.parents, comb), 1):
                    parent_prob = probabilities.get(parent_id, 0.0)
                    parent = network[parent_id]
                    operation = "P" if state == 1 else "1-P"
                    value = parent_prob if state == 1 else (1 - parent_prob)
                    
                    f.write(f"  Родитель {j}: {parent.name} (ID {parent_id})\n")
                    f.write(f"  Состояние: {state} → {operation}({parent_prob:.4f}) = {value:.4f}\n")

                    prob_comb *= value
                    f.write(f"  Текущая prob_comb: {prob_comb:.4f}\n")

                contribution = p_node * prob_comb   
                tr_temp = total
                total = total + contribution
                f.write(f"Вклад комбинации: {p_node:.4f} * {prob_comb:.4f} = {contribution:.4f}\n")
                f.write(f"!!!Накопление суммы влияний: {tr_temp:.4f} + {contribution:.4f} = {total:.4f}\n")
                f.write(f"Накопленный сумма (свертка): {total:.4f}\n")

            probabilities[node_id] = total
            f.write(f"\nИтоговая вероятность: {total:.4f}\n")
            f.write("="*50 + "\n\n")

    return probabilities

def get_conditional_probability(network, node_id, parent_states):
    node = network.get(node_id)
    if not node:
        raise ValueError(f"Узел {node_id} не найден")
    return node.get_probability(parent_states)

        
def save_results(final_probs,
                 graph_data,
                 output= "results.json",
                 debag_file = None):
    grouped_data = {}
    
    # Строим вспомогательные структуры данных
    id_to_node = {n['id']: n for n in graph_data['nodes']}
    child_map = defaultdict(list)
    
    # Важно: убедимся в правильном направлении связей
    for link in graph_data['links']:
        source = link['source']
        target = link['target']
        child_map[source].append(target)  # source -> target (родитель -> ребенок)

    # Отладочная информация
    debug_info = {
        'all_prepare_nodes': [],
        'found_side_effects': defaultdict(list)
    }

    # Обрабатываем все препараты
    for node in graph_data['nodes']:
        if node.get('label') == 'prepare':
            drug_id = node['id']
            drug_name = node['name']
            debug_info['all_prepare_nodes'].append(drug_id)

            # Собираем информацию о препарате
            drug_entry = {
                "drug_id": drug_id,
                "drug_name_en": node.get('name_en', drug_name),
                "side_effects": {}
            }

            # Модифицированная функция сбора эффектов
            def collect_effects(parent_id, path=[]):
                effects = {}
                for child_id in child_map.get(parent_id, []):
                    if child_id not in id_to_node:
                        print(f"⚠️ Не найден узел с ID: {child_id}")
                        continue

                    child = id_to_node[child_id]
                    child_name = child.get('name', 'unnamed')
                    
                    # Записываем путь для отладки
                    current_path = path + [f"{child_name} ({child_id})"]
                    
                    # Если узел помечен как побочный эффект (можем менять условие)
                    if child.get('label') in ['side_effect', 'side_e', 'effect']:
                        prob = round(float(final_probs.get(child_id, 0.0)), 4)
                        effects[child_name] = {
                            "id": child_id,
                            "probability": prob,
                            "path": " → ".join(current_path)  # Для отладки
                        }
                        debug_info['found_side_effects'][drug_id].append(child_id)
                    # Продолжаем обход графа
                    else:
                        nested = collect_effects(child_id, current_path)
                        effects.update(nested)
                return effects

            # Собираем все эффекты
            drug_entry["side_effects"] = collect_effects(drug_id, [f"{drug_name} ({drug_id})"])
            
            # Добавляем в результат
            grouped_data[drug_name] = drug_entry

    # Сохраняем результат
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(grouped_data, f, ensure_ascii=False, indent=4)

    if debag_file:
        # Сохраняем отладочную информацию
        with open(debag_file, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=4)

    print("Сохранение завершено. Проверьте debug_info.json для анализа структуры графа")

if __name__ == "__main__":

    WORK_DIR = "bayes_network"

    # Конфигурационные файлы
    GRAPH_FILE = f'{WORK_DIR}\\spironolacton_lizinopril.json'
    OPT_FILE = f'{WORK_DIR}\\probabilities_opt.json'
    DOSES_FILE = f'{WORK_DIR}\\prepare_doses.json'
    RESULT_FILE = f'{WORK_DIR}\\results.json'
    DEBAG_FILE = f'{WORK_DIR}\\debug_info.json'
    TRACE_FILE = f'{WORK_DIR}\\calculation_trace.txt'

    # Загрузка графа
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    # Шаг 1: Создать базовые вероятности
    # create_base_probabilities(graph)
    
    # Шаг 2: Создать файл доз
    create_doses_file(graph, DOSES_FILE)
    
    # Шаг 3: Загрузить объединенные данные
    prob_data = load_combined_data(
        prob_file=OPT_FILE,         # Явно указываем оптимизированные вероятности
        doses_file=DOSES_FILE       # Берем актуальные дозы
    )
    
    # Построение сети с новыми параметрами
    network = build_network(graph, prob_data)
    
    # Перерасчет вероятностей
    final_probs = calculate_probabilities(network, TRACE_FILE)
    
    # Сохранение обновленных результатов
    save_results(final_probs, graph, RESULT_FILE, DEBAG_FILE)
    
    # # Построение и расчет сети
    # network = build_network(graph, prob_data)
    # final_probs = calculate_probabilities(network)
    
    # # Сохранение результатов
    # save_results(final_probs, graph, "results.json")
    
    print("Расчеты успешно завершены. Результаты в results.json")