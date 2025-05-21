import networkx as nx
import os
from process_nx_graph import ProcessNxGraph
import json

DIR_GRAPHS = "process_yEd_graph\\data\\graph_yEd_verified_"
processor = ProcessNxGraph()
graphs=[]

# Загрузка словаря побочек
FILENAME_SIDE_E_DICT = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"
with open(FILENAME_SIDE_E_DICT, "r", encoding="utf-8") as file:
    side_e_dict = json.load(file)[0]

for i in range(2):
    cur_list = f"{DIR_GRAPHS}{i+1}"
    for filename in os.listdir(cur_list):
        # print(filename)
        G = processor.load_graphml(f"{cur_list}\\{filename}")
        G = processor.concat_hanging_act_mech(G)
        G = processor.remove_remaining_noun(G)
        G = processor.link_isolated_side_e(G)
        G = processor.convert_side_e_by_dict(G, side_e_dict)

        # Изолированные вершины
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)

        # Ищем все вершины с label="A" и label="B"
        label_A = "систолический"
        label_B = "ад"
        nodes_A = [n for n, attr in G.nodes(data=True) if f"{label_A}" in attr.get("label")]
        nodes_B = [n for n, attr in G.nodes(data=True) if f"{label_B}" in attr.get("label")]

        # Проверяем, есть ли ребро между любой вершиной "A" и любой вершиной "B"
        has_AB_edge = any(G.has_edge(a, b) for a in nodes_A for b in nodes_B)
        has_BA_edge = any(G.has_edge(a, b) for a in nodes_B for b in nodes_A)

        if has_AB_edge:
            print(f"В графе {filename} есть ребро между вершинами с label='{label_A}' и label='{label_B}'")
        if has_BA_edge:
            print(f"В графе {filename} есть ребро между вершинами с label='{label_B}' и  label='{label_A}'")

        graphs.append(G)

merged_graphs = processor.merge_graph_list_by_label(graphs)
print(merged_graphs)

try:
    cycle = nx.find_cycle(merged_graphs, orientation="original")
    print("Найден цикл:", cycle)
    for target, source, _ in cycle:
        print("target:", processor.extract_label_and_tag(merged_graphs, target),
              "source:", processor.extract_label_and_tag(merged_graphs, source))
except nx.NetworkXNoCycle:
    print("Граф ациклический")

merged_graphs_json = processor.graph2json(merged_graphs)
with open("process_yEd_graph\\data\\merged_graphs\\all_graphs.json", 'w', encoding='utf-8') as f:
    json.dump(merged_graphs_json, f, indent=4, ensure_ascii=False)


nx.write_gexf(merged_graphs, "process_yEd_graph\\data\\merged_graphs\\all_graphs.gexf", version="1.2draft")