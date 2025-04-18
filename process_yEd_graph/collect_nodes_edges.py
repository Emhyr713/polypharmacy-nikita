
import os
import csv
from process_nx_graph import ProcessNxGraph
from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer
custom_morph = EnhancedMorphAnalyzer()

if __name__ == "__main__":
    dir_graph_edit = "process_yEd_graph\\data\\graph_yEd_raw"
    dir_graph_processed = "process_yEd_graph\\data\\graph_yEd_processed"

    DIR_CSV_NODES = "process_yEd_graph\\data\\list_nodes.csv"
    DIR_CSV_EDGES = "process_yEd_graph\\data\\list_edges.csv"
    

    nodes_set = set()
    edges_set = set()

    processor = ProcessNxGraph(custom_morph)


    for i_dir in range (2):
        current_dir_read = f"{dir_graph_edit}_{i_dir+1}"
        current_dir_save = f"{dir_graph_processed}_{i_dir+1}"
        for i, filename in enumerate(os.listdir(current_dir_read)):
            if filename.endswith(".graphml"):

                print("filename opened:", filename)

                # Загрузка графа
                G = processor.yEd2graph(f"{current_dir_read}\\{filename}")
                # # Этап 1
                # G = processor.concat_hanging_act_mech(G)
                # # Этап 2
                # G = processor.remove_remaining_noun(G)
                # Подсчёт вершин и рёбер
                processor.collect_nodes_edges(G)

                # xml_str = graph2yEd(G, loaded_yed=True)
                # with open(f"{current_dir_save}\\{filename}", "w", encoding="utf-8") as f:
                #     f.write(xml_str)


    # Сохранение в csv для узлов
    header_nodes = ["label", "tag"]
    with open(DIR_CSV_NODES, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")   # Указываем delimiter при создании writer
        writer.writerow(header_nodes)           # Записываем заголовки
        writer.writerows(processor.nodes_set)             # Записываем кортежи

    # Сохранение в csv для рёбер
    header_edges = ["source", "target"]  # Пример для рёбер
    with open(DIR_CSV_EDGES, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")   # Указываем delimiter при создании writer
        writer.writerow(header_edges)           # Записываем заголовки
        writer.writerows(processor.edges_set)             # Записываем кортежи

    # print_message(f"nodes_set:{len(nodes_set)}, edges_set:{len(edges_set)}")