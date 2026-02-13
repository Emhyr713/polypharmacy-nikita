 
import json
import os
import sys
sys.path.append("")

from process_yEd_graph.process_nx_graph import ProcessNxGraph

def get_list_uniq_side_e(json_data):
    uniq_set = set()

    for drug, content in json_data.items():
        for side_e in content['side_e_parts']:
            uniq_set.add(side_e)

    return uniq_set

if __name__ == "__main__":
    filename = "make_side_effect_dataset\\data\\sef_dataset.json"
    with open(filename, "r", encoding="utf-8") as file:
        json_dataset = json.load(file)

    uniq_list = get_list_uniq_side_e(json_dataset)

    # Словарь синонимов
    side_e_synonim_dict_filename = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"
    with open(side_e_synonim_dict_filename, "r", encoding="utf-8") as file:
        side_synonim_dict = json.load(file)[0]
        writed_side_e = {
            item
            for key, side_e_list in side_synonim_dict.items()
            for item in [key] + side_e_list  # Добавляем ключ и все элементы списка
        }

    print(len(writed_side_e))

    DIR_GRAPHS_OUTPUT_104 = "process_yEd_graph\\data\\graph_yEd_with_side_e_from_dataset_"
    process_graph = ProcessNxGraph(morph_analyzer=None)

    side_e_in_graphs = []

    for i in range(2):
        currunt_dir = f"{DIR_GRAPHS_OUTPUT_104}{i+1}"
        for filename in os.listdir(currunt_dir):
            if not filename.endswith(".graphml"):
                continue

            print("filename:", filename)

            # Загрузка графа из файла
            G_yEd = process_graph.load_graphml(f"{currunt_dir}\\{filename}")

            # Нахождение существующих побочек в графе
            side_e_nodes = process_graph.find_node_by_tag(G_yEd, find_tag="side_e")
            side_e_in_graphs.extend([process_graph.extract_label_and_tag(G_yEd, node)[0]
                               for node in side_e_nodes])
            
            
    print("len(side_e_in_graphs)", len(set(side_e_in_graphs)))
    # print("len(side_e_in_graphs-writed_side_e)", len(side_e_in_graphs-writed_side_e))
            
    uniq_list = (uniq_list | set(side_e_in_graphs)) - writed_side_e
    filepath = "make_side_effect_dataset\\data\\sef_uniq_list_3.txt"
    with open(filepath, "w", encoding="utf-8") as file:
        file.write("\n".join(sorted(uniq_list)))

    filepath = "make_side_effect_dataset\\data\\all_side_e.txt"
    with open(filepath, "w", encoding="utf-8") as file:
        file.write("\n".join(sorted((uniq_list | set(side_e_in_graphs) | writed_side_e))))