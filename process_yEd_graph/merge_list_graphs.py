import networkx as nx
import os
import json
from process_nx_graph import ProcessNxGraph

# DIR_GRAPHS = "process_yEd_graph\\data\\graph_yEd_processed_"
DIR_GRAPHS = "process_yEd_graph\\data\\graph_yEd_processed_104_"
processor = ProcessNxGraph()
graphs=[]

GRAPHS_NAME =  [
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
               ]

# Загрузка словаря побочек
FILENAME_SIDE_E_DICT = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"
with open(FILENAME_SIDE_E_DICT, "r", encoding="utf-8") as file:
    side_e_dict = json.load(file)[0]

for i in range(2):
    cur_list = f"{DIR_GRAPHS}{i+1}"
    for filename in os.listdir(cur_list):
        print(filename)
        if filename in GRAPHS_NAME:
            G = processor.load_graphml(f"{cur_list}\\{filename}")
            G = processor.concat_hanging_act_mech(G)
            G = processor.remove_remaining_noun(G)
            G = processor.link_isolated_side_e(G)
            G = processor.convert_side_e_by_dict(G, side_e_dict)
            graphs.append(G)

merged_graphs = processor.merge_graph_list_by_label(graphs)
print(merged_graphs)

merged_graphs_json = processor.graph2json(merged_graphs)
with open("process_yEd_graph\\data\\merged_graphs\\10_graphs.json", 'w', encoding='utf-8') as f:
    json.dump(merged_graphs_json, f, indent=4, ensure_ascii=False)

nx.write_gexf(merged_graphs, "process_yEd_graph\\data\\merged_graphs\\10_graphs.gexf", version="1.2draft")