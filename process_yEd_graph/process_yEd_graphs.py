import uuid
import json
import os
from xml.sax.saxutils import escape

from process_nx_graph import ProcessNxGraph

if __name__ == "__main__":

    # Инициализация директорий и файлов
    DIR_GRAPHS_INPUT = "process_yEd_graph\\data\\graph_yEd_raw_"
    DIR_GRAPHS_OUTPUT = "process_yEd_graph\\data\\graph_yEd_with_side_e_from_dataset_"
    DIR_GRAPHS_OUTPUT_PROCESSED = "process_yEd_graph\\data\\graph_yEd_processed_"
    DIR_GRAPHS_OUTPUT_104 = "process_yEd_graph\\data\\graph_yEd_processed_104_"
    FILENAME_SIDE_E_DATASET = "make_side_effect_dataset\\data\\sef_dataset.json"
    FILENAME_SIDE_E_DICT = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"


    # Смещение по оси х при добавлении в граф вершины
    X_SHIFT = 200

    process_graph = ProcessNxGraph()

    # Загрузка датасета побочек
    with open(FILENAME_SIDE_E_DATASET, "r", encoding="utf-8") as file:
        side_e_dataset = json.load(file)

    # Загрузка словаря побочек
    with open(FILENAME_SIDE_E_DICT, "r", encoding="utf-8") as file:
        side_e_dict = json.load(file)[0]
    map_side_e_dict = {side_e:side_e_104
                       for side_e_104, side_e_list in side_e_dict.items()
                       for side_e in side_e_list}

    for i in range(2):
        currunt_dir_input = f"{DIR_GRAPHS_INPUT}{i+1}"
        currunt_dir_output = f"{DIR_GRAPHS_OUTPUT}{i+1}"
        currunt_dir_output_104 = f"{DIR_GRAPHS_OUTPUT_104}{i+1}"
        currunt_dir_output_processed = f"{DIR_GRAPHS_OUTPUT_PROCESSED}{i+1}"

        for filename in os.listdir(currunt_dir_input):
            if not filename.endswith(".graphml"):
                continue

            # Загрузка графа из файла
            G_yEd = process_graph.load_graphml(f"{currunt_dir_input}\\{filename}")
            
            # Нахождение узлов -- препаратов и вычисление точки создания узлов
            prepare_nodes = process_graph.find_prepare_nodes(G_yEd)
            drug_name = '+'.join(sorted(process_graph.extract_label_and_tag(G_yEd, node)[0]
                                        for node in prepare_nodes
                                        )
                                )
            x = max(G_yEd.nodes[node].get("x", 0) for node in prepare_nodes)
            y = min(G_yEd.nodes[node].get("y", 0) for node in prepare_nodes)

            # Нахождение существующих побочек в графе
            side_e_nodes = process_graph.find_node_by_tag(G_yEd, find_tag="side_e")
            side_e_in_graph = [process_graph.extract_label_and_tag(G_yEd, node)[0]
                               for node in side_e_nodes]
            
            print(filename, len(side_e_nodes), len(side_e_in_graph))

            # Поиск побочек в соответсвии с лекарством
            side_e_load = side_e_dataset.get(drug_name, {}).get("side_e_parts", [])

            # Формирование списка побочек для добавления в граф
            need_to_add_side = [s for s in side_e_load if s not in side_e_in_graph]

            # Добавление побочек на граф со смещением относительно препаратов
            if side_e_load:
                for side_e in need_to_add_side:
                    G_yEd.add_node(str(uuid.uuid4()),
                                label= f"{side_e}(side_e)",
                                x=str(float(x)+X_SHIFT), y=y
                            )
            else:
                print(f"Не найден датасет побочек: {filename}, {drug_name}")

            # Объединение дублирующихся вершин
            G_yEd = process_graph.merge_nodes_by_label(G_yEd)
            
            # Сохранение в формате yEd
            process_graph.save_nxGraph_to_yEd(G_yEd, f"{currunt_dir_output}\\{filename}")


            # Устранение "служебных" висячих узлов с тегом "action"
            G_yEd = process_graph.concat_hanging_act_mech(G_yEd)
            # Устранение "служебных" noun
            G_yEd = process_graph.remove_remaining_noun(G_yEd)

            process_graph.save_nxGraph_to_yEd(G_yEd, f"{currunt_dir_output_processed}\\{filename}")

            # Преобразование согласно словарю
            side_e_nodes = process_graph.find_node_by_tag(G_yEd, find_tag="side_e")
            for node in side_e_nodes:
                label, _ = process_graph.extract_label_and_tag(G_yEd, node)
                new_label = map_side_e_dict.get(label, label)
                G_yEd.nodes[node]["label"] = f"{new_label}(side_e)"

            # Объединение дублирующихся вершин
            G_yEd = process_graph.merge_nodes_by_label(G_yEd)
            process_graph.save_nxGraph_to_yEd(G_yEd, f"{currunt_dir_output_104}\\{filename}")