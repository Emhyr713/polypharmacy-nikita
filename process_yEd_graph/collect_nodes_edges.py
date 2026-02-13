
import os
import csv

from process_nx_graph import ProcessNxGraph
from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer
custom_morph = EnhancedMorphAnalyzer()

def filter_set_node_edge(node_set, edge_set):
    # Определяем нужные теги
    allowed_tags = {"action", "mechanism", "side_e"}

    # Фильтруем список кортежей
    edge_set = set([
        (source, source_tag, target, target_tag)
        for source, source_tag, target, target_tag in edge_set
        if source_tag in allowed_tags and target_tag in allowed_tags
    ])
    return node_set, edge_set

if __name__ == "__main__":
    # dir_graph_edit = "process_yEd_graph\\data\\graph_yEd_raw"
    dir_graph_edit = "process_yEd_graph\\data\\graph_yEd_verified"

    # dir_graph_processed = "process_yEd_graph\\data\\graph_yEd_processed"
    # dir_graph_processed = "process_yEd_graph\\data\\graph_yEd_verified"

    CSV_NODES_FILENAME = "process_yEd_graph\\data\\list_nodes_verified_folder.csv"
    CSV_EDGES_FILENAME = "process_yEd_graph\\data\\list_edges_verified_folder.csv"
    
    processor = ProcessNxGraph(custom_morph)

    for i_dir in range (2):
        current_dir_read = f"{dir_graph_edit}_{i_dir+1}"
        # current_dir_save = f"{dir_graph_processed}_{i_dir+1}"
        for i, filename in enumerate(os.listdir(current_dir_read)):
            if filename.endswith(".graphml"):
                print("filename opened:", filename)

                # Загрузка графа
                G = processor.load_graphml(f"{current_dir_read}\\{filename}")

                # # Этап 1
                G = processor.concat_hanging_act_mech(G)
                # # Этап 2
                G = processor.remove_remaining_noun(G)

                # Подсчёт вершин и рёбер
                processor.collect_nodes_edges(G, join_tag=False)

    node_set, edge_set = filter_set_node_edge(processor.nodes_set,
                                              processor.edges_set)

    # Сохранение в csv для узлов
    header_nodes = ["label", "tag"]
    with open(CSV_NODES_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")   
        writer.writerow(header_nodes)          
        writer.writerows(node_set)

    # Сохранение в csv для рёбер
    header_edges = ["source","source_tag","target","target_tag"]
    with open(CSV_EDGES_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")   
        writer.writerow(header_edges)
        writer.writerows(edge_set)
