import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph


def topological_sort_pure_by_predecessors(graph):
    """
    Выполняет топологическую сортировку ориентированного графа.
    
    Параметры:
    ----------
    graph : dict
        Словарь предшественников: ключ — узел, значение — список предков.
        Пример: {'C': ['A', 'B'], 'B': ['A'], 'A': []}
    
    Возвращает:
    ----------
    list
        Список узлов в порядке топологической сортировки (по поколениям).
    
    Исключения:
    ----------
    ValueError
        Если граф содержит цикл.
    """
    # Собираем все узлы
    all_nodes = set(graph.keys())
    for predecessors in graph.values():
        all_nodes.update(predecessors)
    
    # Входящая степень — это длина списка предков
    indegree = {node: len(graph.get(node, [])) for node in all_nodes}
    
    # Строим обратное отображение: для каждого узла — его потомки
    successors = {node: [] for node in all_nodes}
    for node, preds in graph.items():
        for pred in preds:
            successors[pred].append(node)
    
    # Узлы с нулевой входящей степенью
    zero_indegree = [node for node in all_nodes if indegree[node] == 0]
    
    result = []
    while zero_indegree:
        this_generation = zero_indegree
        zero_indegree = []
        for node in this_generation:
            result.append(node)
            for child in successors[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    zero_indegree.append(child)

    if any(deg > 0 for deg in indegree.values()):
        raise ValueError("Граф содержит цикл и не может быть топологически отсортирован.")
    
    return result
    


def topological_sort(nodes_to_sort, parent_map):
    """
    Выполняет топологическую сортировку узлов.
    Это нужно, чтобы вычислять вероятности в правильном порядке: от "предков" к "потомкам".
    """
    from collections import defaultdict # Для удобного создания словарей со значениями по умолчанию

    print("parent_map:", parent_map)
    print("nodes_to_sort:", nodes_to_sort)

    in_degree = {nid: 0 for nid in nodes_to_sort}
    children_map = defaultdict(list)
    for child, parents in parent_map.items():
        if child in nodes_to_sort:
            for parent in parents:
                if parent in nodes_to_sort:
                    in_degree[child] += 1
                    children_map[parent].append(child)

    
    queue = [nid for nid in nodes_to_sort if in_degree[nid] == 0]
    sorted_list = []
    while queue:
        u = queue.pop(0)
        sorted_list.append(u)
        for v in children_map[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return sorted_list



if __name__ == "__main__":
    # graph_path = 'bayes_network\\data\\graphs\\TEST.json'
    graph_path = 'bayes_network\\data\\graphs\\graphs_10_4.json'
    # graph_path = 'bayes_network\\data\\graphs\\TEST.json'
    # graph_path = 'bayes_network\\data\\graphs\\TEST.json'

    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_from_json = json.load(f)
    # print("graph_from_json['nodes']", graph_from_json['nodes'])

    # Библиотека
    G = json_graph.node_link_graph(graph_from_json)
    lib_topological_sort = list(nx.topological_sort(G))
    # print("lib_topological_sort:", lib_topological_sort, "\n\n")

    # Переписанная из библиотеки
    parent_map = {node['id']: node['parents'] for node in graph_from_json['nodes']}
    # print("parent_map:", parent_map, "\n\n")
    lib_writed_topological_sort = topological_sort_pure_by_predecessors(parent_map)
    # print("lib_writed_topological_sort:", lib_writed_topological_sort, "\n\n")

    # Женя
    nodes_to_sort = [node['id'] for node in graph_from_json['nodes']]
    parent_map = {node['id']: node['parents'] for node in graph_from_json['nodes']}
    jenya_topological_sort = topological_sort(nodes_to_sort, parent_map)
    # print("jenya_topological_sort:", jenya_topological_sort, "\n\n")


    print("lib_topological_sort == lib_writed_topological_sort", lib_topological_sort == lib_writed_topological_sort, "\n\n")
    print("lib_topological_sort == jenya_topological_sort", lib_topological_sort == jenya_topological_sort, "\n\n")
    print("lib_writed_topological_sort == jenya_topological_sort", lib_writed_topological_sort == jenya_topological_sort, "\n\n")
    
