import networkx as nx
import os
import json
from process_nx_graph import ProcessNxGraph

DIR_GRAPHS = "process_yEd_graph\\data\\graph_yEd_verified_1"
processor = ProcessNxGraph()

FILENAME = "process_yEd_graph\\data\\graph_yEd_verified_1\\drug_trimetatizin.graphml"


# Загрузка словаря побочек
FILENAME_SIDE_E_DICT = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"
with open(FILENAME_SIDE_E_DICT, "r", encoding="utf-8") as file:
    side_e_dict = json.load(file)[0]

def single_graph(filename):
    G = processor.load_graphml(filename)
    G = processor.concat_hanging_act_mech(G)
    G = processor.remove_remaining_noun(G)
    G = processor.link_isolated_side_e(G)
    G = processor.convert_synonim_word_by_dict(G, side_e_dict, find_tag='side_e')

    return G

graph = single_graph(FILENAME)

graph = processor.graph2json(graph)
with open("process_yEd_graph\\data\\test_graphs\\drug_trimetatizin.json", 'w', encoding='utf-8') as f:
    json.dump(graph, f, indent=4, ensure_ascii=False)
