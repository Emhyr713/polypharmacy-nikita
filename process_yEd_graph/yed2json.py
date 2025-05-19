import json
from process_nx_graph import ProcessNxGraph

processor = ProcessNxGraph()

GRAPH_LIST_FILENAME = [
    "process_yEd_graph\\data\\graph_yEd_linked_side_e_1\\drug_lizinopril.graphml",
    "process_yEd_graph\\data\\graph_yEd_linked_side_e_1\\drug_spironolakton.graphml"
]

graphs = []

for filename in GRAPH_LIST_FILENAME:
    G = processor.load_graphml(filename)
    G = processor.link_isolated_side_e(G)
    graphs.append(G)

merged_graphs = processor.merge_graph_list_by_label(graphs)
merged_graphs_json = processor.graph2json(merged_graphs)

processor.plot_graph(merged_graphs)

FILE_SAVE = "process_yEd_graph\\data\\merged_graphs\\spironolacton_lizinopril.json"
with open(FILE_SAVE, 'w', encoding='utf-8') as f:
    json.dump(merged_graphs_json, f, indent=4, ensure_ascii=False)
