import networkx as nx
import os
import json
import pandas as pd

from process_nx_graph import ProcessNxGraph
from simplification_graph import simplification_graph
# from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer

# Инициализация морфологического анализатора
# custom_morph = EnhancedMorphAnalyzer()

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
OUTPUT_JSON = "process_yEd_graph\\data\\merged_graphs\\graph_all.json"
OUTPUT_GEXF = "process_yEd_graph\\data\\merged_graphs\\graph_all.gexf"
MODEL_PATH = "train_synonim_model/data/synonym-model_3"


def load_side_effect_dict(filepath):
    """Загружает словарь побочных эффектов из JSON-файла."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)[0]


def load_and_process_graphs(dir_base, graph_names, side_e_dict, processor):
    """
    Загружает и предварительно обрабатывает графы из указанных директорий.
    Возвращает список обработанных графов.
    """
    graphs = []
    for i in range(2):
        cur_dir = f"{dir_base}{i + 1}"
        for filename in os.listdir(cur_dir):
            # if filename in graph_names:
            filepath = os.path.join(cur_dir, filename)
            G = processor.load_graphml(filepath)
            G = processor.concat_hanging_act_mech(G)
            G = processor.remove_remaining_noun(G)
            # G = processor.link_isolated_side_e(G)
            G = processor.convert_synonim_word_by_dict(G, side_e_dict, 'side_e')
            graphs.append(G)
            print(f"Загружен и обработан: {filename} (узлов: {G.number_of_nodes()}, рёбер: {G.number_of_edges()})")
    return graphs


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
    G = simplification_graph(edges_df, model_path)
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
                if tag != 'side_e':
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
    side_e_dict = load_side_effect_dict(FILENAME_SIDE_E_DICT)

    # Загрузка и предварительная обработка графов
    graphs = load_and_process_graphs(DIR_GRAPHS_BASE, GRAPHS_NAME, side_e_dict, processor)

    # Объединение графов
    merged_graphs = processor.merge_graph_list_by_label(graphs)
    print(f"Объединённый граф: узлов={merged_graphs.number_of_nodes()}, рёбер={merged_graphs.number_of_edges()}")

    # Упрощение 1: Удаление побочек, не входящих в словарь
    merged_graphs = simplify_graph_by_side_e_dict(merged_graphs, side_e_dict, processor)

    # Упрощение 2: Применение ML-модели для синонимов
    merged_graphs = simplify_graph_by_model(merged_graphs, MODEL_PATH, processor)

    # Удаление всех листьев, кроме побочек
    merged_graphs = remove_leaf_nodes_cascade_except_side_e(merged_graphs, processor)

    # Проверка на циклы
    processor.check_graph_cycles(merged_graphs)

    # Сохранение результатов
    save_graph_outputs(merged_graphs, OUTPUT_JSON, OUTPUT_GEXF, processor)


if __name__ == "__main__":
    main()