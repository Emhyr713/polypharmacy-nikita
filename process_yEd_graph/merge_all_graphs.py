import networkx as nx
import os
from process_nx_graph import ProcessNxGraph

# DIR_GRAPHS = "process_yEd_graph\\data\\graph_yEd_processed_"
DIR_GRAPHS = "process_yEd_graph\\data\\graph_yEd_processed_104_"
processor = ProcessNxGraph()
graphs=[]

for i in range(2):
    cur_list = f"{DIR_GRAPHS}{i+1}"
    for filename in os.listdir(cur_list):

        G = processor.load_graphml(f"{cur_list}\\{filename}")
        G = processor.link_isolated_side_e(G)
        graphs.append(G)

merged_graphs = processor.merge_graph_list_by_label(graphs)
nx.write_gexf(merged_graphs, "process_yEd_graph\\data\\merged_graphs\\all_drugs_104.gexf", version="1.2draft")