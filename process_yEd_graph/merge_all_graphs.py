import networkx as nx
import os
from process_nx_graph import ProcessNxGraph
import json

def load_side_effect_dict(filename):
    """Загрузка словаря побочных эффектов"""
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)[0]
    
def load_synonim_dict(filename):
    """
    Преобразует структуру кластеров в словарь вида {"первый элемент": [весь список]}
    
    Args:
        json_data (dict): Исходные данные в формате JSON
        
    Returns:
        dict: Преобразованный словарь
    """

    with open(filename, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    transformed = {}
    
    for cluster in json_data["clusters"].values():
        if cluster["labels"]:  # Проверяем, что список не пустой
            first_element = cluster["labels"][0]
            transformed[first_element] = cluster["labels"]

    # print("transformed:", transformed)
    
    return transformed

def process_graph_files(directory_prefix, num_graphs=2):
    """Обработка всех файлов графов в указанных директориях.
    Возвращает:
        - список графов (graphs)
        - словарь {название узла: [список графов, где он встречается]}
    """
    processor = ProcessNxGraph()
    graphs = []
    node_to_graphs = {}  # Словарь для хранения узлов и графов
    
    for i in range(num_graphs):
        current_dir = f"{directory_prefix}{i+1}"
        for filename in os.listdir(current_dir):
            graph_name = os.path.splitext(filename)[0]  # Имя графа без расширения
            G = processor.load_graphml(f"{current_dir}\\{filename}")
            G = processor.concat_hanging_act_mech(G)
            G = processor.remove_remaining_noun(G)
            G = processor.link_isolated_side_e(G)
            
            # Удаление изолированных вершин
            isolated_nodes = list(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)
            
            graphs.append(G)
            
            # Заполняем словарь node_to_graphs
            for node in G.nodes():
                node_label, node_tag = processor.extract_label_and_tag(G, node)
                if node_label not in node_to_graphs:
                    node_to_graphs[node_label] = []
                node_to_graphs[node_label].append(graph_name)
    
    return graphs, node_to_graphs

def check_specific_labels(graph, filename, label_A, label_B):
    """Проверка наличия ребра между вершинами с определенными метками"""
    nodes_A = [n for n, attr in graph.nodes(data=True) if f"{label_A}" in attr.get("label")]
    nodes_B = [n for n, attr in graph.nodes(data=True) if f"{label_B}" in attr.get("label")]

    has_AB_edge = any(graph.has_edge(a, b) for a in nodes_A for b in nodes_B)
    has_BA_edge = any(graph.has_edge(a, b) for a in nodes_B for b in nodes_A)

    if has_AB_edge:
        print(f"В графе {filename} есть ребро между вершинами с label='{label_A}' и label='{label_B}'")
    if has_BA_edge:
        print(f"В графе {filename} есть ребро между вершинами с label='{label_B}' и label='{label_A}'")

def check_graph_cycles(graph, node_to_graphs, synonim_dict):
    """Проверяет, есть ли в графе циклы. Возвращает True/False."""
    # print("synonim_dict:", synonim_dict)
    # print("node_to_graphs", node_to_graphs)
    try:
        cycle = nx.find_cycle(graph, orientation="original")
        print("Найден цикл:", cycle)
        processor = ProcessNxGraph()  # Предполагается, что этот класс определён
        for target, source, _ in cycle:
            target_label, target_tag = processor.extract_label_and_tag(graph, target)
            source_label, source_tag = processor.extract_label_and_tag(graph, source)
            # print(f"synonim_dict[{target_label}]:{synonim_dict[target_label]}")
            # print(f"synonim_dict[{source_label}]:{synonim_dict[source_label]}")
            print(f"target: {target_label}({target_tag}) - {[node_to_graphs.get(word) for word in synonim_dict[target_label]]}\n",
                  f"source: {source_label}({source_tag}) - {[node_to_graphs.get(word) for word in synonim_dict[source_label]]}\n\n")
            
        return True  # Цикл найден
    except nx.NetworkXNoCycle:
        print("Граф ациклический")
        return False  # Циклов нет

def save_graph_data(processor, graph, json_path, gexf_path):
    """Сохранение графа в JSON и GEXF форматах"""
    merged_graphs_json = processor.graph2json(graph)

    if merged_graphs_json:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_graphs_json, f, indent=4, ensure_ascii=False)
    else:
        print('Не удалось сохранить в json')
    
    nx.write_gexf(graph, gexf_path, version="1.2draft")

if __name__ == "__main__":
    # Конфигурационные параметры
    DIR_GRAPHS = "process_yEd_graph\\data\\graph_yEd_verified_"
    FILENAME_SIDE_E_DICT = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"
    FILENAME_SYNONIM_DICT = "process_yEd_graph\\data\\synonim_dict.json"
    OUTPUT_JSON = "process_yEd_graph\\data\\merged_graphs\\all_graphs.json"
    OUTPUT_GEXF = "process_yEd_graph\\data\\merged_graphs\\all_graphs.gexf"

    processor = ProcessNxGraph()

    tag_list = processor.get_tags_list()
    tag_list.remove('side_e')
    print('tag_list', tag_list)
    
    # Основной процесс
    side_e_dict = load_side_effect_dict(FILENAME_SIDE_E_DICT)
    graphs, node_to_graphs = process_graph_files(DIR_GRAPHS)
    merged_graphs = processor.merge_graph_list_by_label(graphs)

    synonim_dict = load_synonim_dict(FILENAME_SYNONIM_DICT)

    merged_graphs = processor.convert_synonim_word_by_dict(merged_graphs, side_e_dict, find_tag='side_e')
    merged_graphs = processor.convert_synonim_word_by_dict(merged_graphs, synonim_dict, find_tag=tag_list)

    
    print(merged_graphs)
    check_graph_cycles(merged_graphs, node_to_graphs, synonim_dict)
    save_graph_data(processor, merged_graphs, OUTPUT_JSON, OUTPUT_GEXF)