import networkx as nx
import sys
sys.path.append("")

from utils import yedLib

def graph2yEd(graph, loaded_yed = False):
    graph_yed = yedLib.Graph()
    # Словарь для хранения идентификаторов узлов
    node_ids = {}
    node_set = set()

    # Добавление узлов
    for i, node in enumerate(graph.nodes()):

        if loaded_yed:
            label = graph.nodes[node].get("label", str(node))
            node_id = label
        else:
            name = graph.nodes[node].get("name", str(node))
            label = graph.nodes[node].get("label", str(node))
            node_id = f"{name}({label})"

        node_ids[node] = node_id
        if node_id not in node_set:
            graph_yed.add_node(node_id, shape="ellipse")
            node_set.add(node_id)
        else:
            print(f"Узел {node_id} уже существует, добавление пропущено.")


    # Добавление рёбер
    for i, (source, target, edge_data) in enumerate(graph.edges(data=True)):
        graph_yed.add_edge(node_ids[source], node_ids[target], arrowhead="standard")

    return graph_yed.get_graph()

