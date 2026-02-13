import networkx as nx
import os
import json
import pandas as pd
from collections import defaultdict

# Графы
from process_nx_graph import ProcessNxGraph
from simplification_graph import simplification_graph

# Семантика
from create_graph.SemanticEmbeddingProcessor import SemanticEmbeddingProcessor

# Константы
DIR_GRAPHS_BASE = "process_yEd_graph\\data\\graph_yEd_verified_"
GRAPHS_NAME = [
    "drug_bisoprolol.graphml",
    "drug_apixaban.graphml",
    "drug_escitalopram.graphml",
    "drug_geparin_na.graphml",
    "drug_kaptopril.graphml",
    "drug_levosimendan.graphml",
    "drug_valsartan.graphml",
    "drug_ivabradin.graphml",
    "drug_izosorbida_dinitrat.graphml",
    "drug_spironolakton.graphml"

    # "drug_ranolazin.graphml"
]
FILENAME_SIDE_E_DICT = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"
OUTPUT_JSON = "process_yEd_graph\\data\\merged_graphs\\graphs_90_4.json"
OUTPUT_GEXF = "process_yEd_graph\\data\\merged_graphs\\graphs_90_4.gexf"
MODEL_PATH = "train_synonim_model\\data\\synonym-model_4"
ABBR_MAP_PATH = "create_graph\\data\\abbrev_dict.json"
ORLOV_DATASET = "bayes_network\\data\\Orlov.json"


def load_json(filepath):
    """Загружает словарь побочных эффектов из JSON-файла."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

def load_and_process_graphs(dir_base, graph_names, side_e_dict, processor, prepare_valid_list):
    """
    Загружает и предварительно обрабатывает графы из указанных директорий.
    Возвращает список обработанных графов.
    """
    graphs = []
    for i in range(3):
        cur_dir = f"{dir_base}{i + 1}"
        for filename in os.listdir(cur_dir):
            # if filename in graph_names:
            filepath = os.path.join(cur_dir, filename)
            G = processor.load_graphml(filepath, lemmatize=False)
            G = processor.concat_hanging_act_mech(G)
            G = processor.remove_remaining_noun(G)
            # G = processor.link_isolated_side_e(G)

            # side_e_dict = find_reference_side_e
            # G = processor.convert_synonim_word_by_dict(G, side_e_dict, 'side_e')

            # check_prepare_nodes(G, processor, prepare_valid_list)
            graphs.append(G)
            print(f"Загружен и обработан: {filename} (узлов: {G.number_of_nodes()}, рёбер: {G.number_of_edges()})")
    return graphs

def find_reference_side_e(G, reference_side_e_list, processor):
    """Определяет эталоны с помощью ML для побочных эффектов и объединяет по похожим."""
    print("G до find_reference_side_e:", G)
    # Поиск вершин побочных эффектов 
    node_set, _ = processor.collect_nodes_edges(G, join_tag=False)
    label_side_e_list = [
        label
        for (label, tag) in node_set
        if tag == 'side_e'
    ]
    # Поиск близких к эталону
    semantic_comp = SemanticEmbeddingProcessor(MODEL_PATH,
                                               abbr_dataset_path=ABBR_MAP_PATH)
    semantic_res = semantic_comp.find_similar_terms(label_side_e_list,
                                                    reference_side_e_list,
                                                    0.99, 1)
    # Создание словаря "Эталон":"Исходная строка"
    normalized_dict = defaultdict(list)
    for original_key, matches in semantic_res.items():
        if matches:
            term = matches[0]['term']
            normalized_dict[term].append(original_key)

    # Приведение к эталонам и объединение по похожим именам узлов
    G = processor.convert_synonim_word_by_dict(G, normalized_dict, 'side_e')
    print("G после find_reference_side_e:", G)
    return G

def find_reference_node(G, processor):
    semantic_comp = SemanticEmbeddingProcessor(MODEL_PATH,
                                               abbr_dataset_path=ABBR_MAP_PATH)
        
    # Множество узлов
    node_set, _ = processor.collect_nodes_edges(G, join_tag=False)

    # Порог схожести
    threshold = 0.99

    # Обработка всех групп тегов
    tag_groups = [
        ['mechanism', 'action'],
        ['absorbtion'],
        ['excretion'],
        ['prot_link'],
        ['metabol'],
        ['hormone'],
        ['distribution']
    ]

    for tags in tag_groups:
        label_tag_list = [label for (label, tag) in node_set if tag in tags]
        label_tag_dict = semantic_comp.cluster_similar_strings(label_tag_list, similarity_threshold=threshold)
        G = processor.convert_synonim_word_by_dict(G, label_tag_dict, tags, lemmatize=False)

    return G


def check_prepare_nodes(G, processor, prepare_valid_list):

    prepare_list = [processor.extract_label_and_tag(G=G, node_id=id)[0]
                    for id in processor.find_prepare_nodes(G)]
    
    # Препараты, которые извлеклись, но нет в эталонном списке
    invalid_prepare_list = [prepare for prepare in prepare_list if not prepare in prepare_valid_list]
    # Препараты, которых нет в графе
    not_in_prepare_list = [prepare for prepare in prepare_valid_list if not prepare in prepare_list]

    print(len(prepare_list), "извлечённые из графов (prepare_list):", prepare_list)
    print(len(invalid_prepare_list), "некорретные (invalid_prepare_list):", invalid_prepare_list)
    print(len(not_in_prepare_list), "нет в графе (not_in_prepare_list):", not_in_prepare_list)


def simplify_graph_by_side_e_dict(G, side_e_dict, processor):
    """Удаляет узлы с тегом 'side_e', которых нет в словаре побочных эффектов."""
    node_set, _ = processor.collect_nodes_edges(G, join_tag=False)
    label_tag_list = [
        (label, tag) for (label, tag) in node_set
        if tag == 'side_e' and label not in side_e_dict
    ]
    print(f"Найдено {len(label_tag_list)} побочных эффектов вне словаря для удаления.")
    G = processor.delete_by_label_tag(G, label_tag_list)
    print(f"Граф после удаления побочек вне словаря: узлов={G.number_of_nodes()}, рёбер={G.number_of_edges()}")
    return G


def simplify_graph_by_model(G, model_path, processor):
    """Упрощает граф с использованием ML-модели синонимов."""
    _, edges_set = processor.collect_nodes_edges(G, join_tag=False)
    print(f"Количество рёбер до упрощения модели: {len(edges_set)}")
    edges_df = pd.DataFrame(list(edges_set), columns=["source", "source_tag", "target", "target_tag"])
    G = simplification_graph(edges_df, model_path, 0.99)
    print(f"Граф после упрощения моделью: узлов={G.number_of_nodes()}, рёбер={G.number_of_edges()}")
    return G


def remove_leaf_nodes_cascade_except_side_e(G, processor):
    """
    Рекурсивно удаляет все листовые узлы (out_degree == 0), кроме тех, у которых тег == 'side_e'.
    Повторяет до тех пор, пока есть что удалять.
    """
    total_removed = 0
    iteration = 0

    while True:
        nodes_to_remove = []
        for node in list(G.nodes()):  # копия списка узлов для безопасного удаления
            if G.out_degree(node) == 0:  # лист (нет исходящих рёбер)
                _, tag = processor.extract_label_and_tag(G, node)
                if tag != 'side_e' and tag != 'prepare':
                    nodes_to_remove.append(node)

        if not nodes_to_remove:
            break  # больше ничего удалять не нужно

        G.remove_nodes_from(nodes_to_remove)
        removed_count = len(nodes_to_remove)
        total_removed += removed_count
        iteration += 1
        print(f"Итерация {iteration}: удалено {removed_count} узлов. Всего удалено: {total_removed}")

    print(f"✅ Завершено. Всего удалено {total_removed} узлов.")
    print(f"Граф после удаления листьев - не побочек: узлов={G.number_of_nodes()}, рёбер={G.number_of_edges()}")
    return G


def insert_intermediate_layer(G, processor, N=11, divider=2):
    import uuid
    
    """
    Рекурсивно вставляет промежуточные слои для всех узлов с > N предков.
    Процесс продолжается до тех пор, пока все узлы не будут иметь ≤ N предков.
    
    Args:
        G: Ориентированный граф (nx.DiGraph)
        processor: Объект для извлечения меток узлов
        N: Пороговое количество предков для вставки промежуточного слоя
        divider: Делитель для вычисления количества промежуточных узлов
    
    Returns:
        Модифицированный граф G
    """
    if not isinstance(G, nx.DiGraph):
        raise ValueError("Граф должен быть ориентированным (DiGraph).")
    
    # Флаг, показывающий, были ли внесены изменения
    modified = True
    
    while modified:
        modified = False
        # Копируем узлы для безопасной итерации
        nodes = list(G.nodes())
        
        for node in nodes:
            preds = list(G.predecessors(node))
            if len(preds) <= N:
                continue
            
            # Если у узла > N предков, обрабатываем его
            modified = True
            
            # Вычисляем количество промежуточных узлов
            k = len(preds) // divider
            if k == 0:
                # Если результат деления меньше 1, создаем хотя бы один узел
                k = 1
            
            # Удаляем все входящие рёбра к node
            for pred in preds:
                G.remove_edge(pred, node)
            
            node_label, node_tag = processor.extract_label_and_tag(G, node)
            
            # Создаём k промежуточных узлов
            inter_nodes = []
            for i in range(k):
                new_id = str(uuid.uuid4())
                new_label = f"inter_{node_label}_{i}(inter_{node_tag})"
                G.add_node(new_id, label=new_label)
                G.add_edge(new_id, node)  # соединяем с потомком
                inter_nodes.append(new_id)
            
            # Распределяем предков по промежуточным узлам
            for i, pred in enumerate(preds):
                inter_idx = i % k
                G.add_edge(pred, inter_nodes[inter_idx])
    
    return G


def save_graph_outputs(G, output_json, output_gexf, processor):
    """Сохраняет граф в JSON и GEXF форматах."""
    graph_json = processor.graph2json(G, lemmatize_nodes=False)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(graph_json, f, indent=4, ensure_ascii=False)
    nx.write_gexf(G, output_gexf, version="1.2draft")
    print(f"Граф сохранён в {output_json} и {output_gexf}")


def main():
    # Инициализация процессора
    processor = ProcessNxGraph()

    # Загрузка словаря побочных эффектов
    side_e_dict = load_json(FILENAME_SIDE_E_DICT)[0]
    prepare_valid_list = load_json(ORLOV_DATASET).keys()
    ref_side_e_list = list(side_e_dict.keys())

    # Загрузка и предварительная обработка графов
    graphs = load_and_process_graphs(DIR_GRAPHS_BASE, GRAPHS_NAME, side_e_dict, processor, prepare_valid_list)

    # Объединение графов
    merged_graphs = processor.merge_graph_list_by_label(graphs)
    check_prepare_nodes(merged_graphs, processor, prepare_valid_list)
    print(f"Объединённый граф: узлов={merged_graphs.number_of_nodes()}, рёбер={merged_graphs.number_of_edges()}")

    # Поиск эталонных побочек и дальнейшее объединение
    merged_graphs = find_reference_side_e(merged_graphs, ref_side_e_list, processor)
    print(f"Побочки приведены к эталонам граф: узлов={merged_graphs.number_of_nodes()}, рёбер={merged_graphs.number_of_edges()}")
    check_prepare_nodes(merged_graphs, processor, prepare_valid_list)

    # Упрощение 1: Применение ML-модели для синонимов
    # merged_graphs = simplify_graph_by_model(merged_graphs, MODEL_PATH, processor)
    merged_graphs = find_reference_node(merged_graphs, processor)
    print("Применение ML-модели для синонимов:")
    check_prepare_nodes(merged_graphs, processor, prepare_valid_list)

    # Упрощение 2: Удаление побочек, не входящих в словарь
    merged_graphs = simplify_graph_by_side_e_dict(merged_graphs, side_e_dict, processor)
    print("Удаление побочек, не входящих в словарь:")
    check_prepare_nodes(merged_graphs, processor, prepare_valid_list)

    # Удаление всех листьев, кроме побочек
    merged_graphs = remove_leaf_nodes_cascade_except_side_e(merged_graphs, processor)
    print("Удаление всех листьев, кроме побочек:")
    check_prepare_nodes(merged_graphs, processor, prepare_valid_list)

    # Создание прослоек для узлов с большим количеством предков
    merged_graphs = insert_intermediate_layer(merged_graphs, processor, N=10, divider=5)
    print("Создание прослоек для узлов с большим количеством предков:")
    check_prepare_nodes(merged_graphs, processor, prepare_valid_list)

    node_set, _ = processor.collect_nodes_edges(merged_graphs, join_tag=False)
    label_tag_list_notin = [
        (label, tag) for (label, tag) in node_set
        if tag == 'side_e' and label not in side_e_dict
    ]
    label_tag_list_in = [
        (label, tag) for (label, tag) in node_set
        if tag == 'side_e' and label in side_e_dict
    ]

    print("label_tag_list_notin", label_tag_list_notin)
    print("label_tag_list_in", label_tag_list_in)

    # Проверка на циклы
    processor.check_graph_cycles(merged_graphs)

    print("До удаления изорированных merged_graphs:", merged_graphs)

    # Удаление изолированных вершин
    merged_graphs.remove_nodes_from(list(nx.isolates(merged_graphs)))

    print("После удаления изорированных вершин merged_graphs:", merged_graphs)
    check_prepare_nodes(merged_graphs, processor, prepare_valid_list)

    # Сохранение результатов
    save_graph_outputs(merged_graphs, OUTPUT_JSON, OUTPUT_GEXF, processor)


if __name__ == "__main__":
    main()