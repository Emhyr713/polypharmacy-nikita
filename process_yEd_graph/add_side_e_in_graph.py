import os
import sys
import csv
import json
import re
import xml.etree.ElementTree as ET
from typing import Tuple, List, Set
sys.path.append("")

from utils.yedLib import Graph
# from process_nx_graph import ProcessNxGraph
from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer

# import pandas as pd
# import networkx as nx

# Для кастомного морфоанализатора (не используется, но загружается)
custom_morph = EnhancedMorphAnalyzer()

# # Регистрация пространств имён для yEd/GraphML
# ET.register_namespace("", "http://graphml.graphdrawing.org/xmlns")
# ET.register_namespace("y", "http://www.yworks.com/xml/graphml")
# ns = {
#     "g": "http://graphml.graphdrawing.org/xmlns",
#     "y": "http://www.yworks.com/xml/graphml"
# }


# def split_by_bracket(text: str) -> Tuple[str, str] | Tuple[None, None]:
#     """
#     Делит строку по скобке (например, 'value (tag)' -> ('value', 'tag'))
#     """
#     match = re.search(r"^(.*)\s*\(([^)]+)\)$", text)
#     return (match.group(1).strip(), match.group(2).strip()) if match else (None, None)


# def load_graphml_to_graph(path: str) -> Graph:
#     """
#     Загружает .graphml файл и возвращает объект Graph с узлами и рёбрами
#     """
#     with open(path, 'r', encoding='utf-8') as f:
#         tree = ET.parse(f)
#     root = tree.getroot()
#     graph_el = root.find("g:graph", ns)

#     g = Graph(directed=graph_el.get("edgedefault"), graph_id=graph_el.get("id"))

#     for node_el in graph_el.findall("g:node", ns):
#         node_id = node_el.get("id")
#         data_el = node_el.find("g:data[y:ShapeNode]", ns)

#         if data_el is None:
#             g.add_node(node_id)
#             continue

#         shape_el = data_el.find("y:ShapeNode", ns)
#         geom = shape_el.find("y:Geometry", ns)

#         x = float(geom.get("x", "0")) if geom is not None else 0.0
#         y = float(geom.get("y", "0")) if geom is not None else 0.0
#         w = geom.get("width") if geom is not None else None
#         h = geom.get("height") if geom is not None else None

#         label_el = shape_el.find("y:NodeLabel", ns)
#         label = label_el.text if label_el is not None else node_id

#         g.add_node(node_id,
#                    label=label,
#                    x=str(x), y=str(y),
#                    shape="ellipse",
#                    width=w, height=h)

#     for edge_el in graph_el.findall("g:edge", ns):
#         g.add_edge(edge_el.get("source"), edge_el.get("target"))

#     return g


# def find_prepare_nodes(G: Graph) -> Tuple[str, Tuple[float, float]]:
#     """
#     Находит все препараты в графе по слову 'prepare' и определяет координаты добавления побочек
#     """
#     found = []

#     for node_id, node in G.nodes.items():
#         for label in node.get_label_text_list():
#             if "prepare" in label.lower():
#                 value, _ = split_by_bracket(label.lower())
#                 found.append((value, float(node.geom["x"]), float(node.geom["y"])))

#     if not found:
#         raise ValueError("No 'prepare' nodes found in graph.")

#     labels, xs, ys = zip(*found)
#     drug_name = '+'.join(sorted(labels))
#     return drug_name, (max(xs), min(ys))


# def find_side_effects(G: Graph) -> Set[str]:
#     """
#     Ищет уже существующие в графе побочные эффекты
#     """
#     side_effects = set()

#     for node_id, node in G.nodes.items():
#         for label in node.get_label_text_list():
#             if "side_e" in label.lower():
#                 value, _ = split_by_bracket(label.lower())
#                 if value:
#                     side_effects.add(value)

#     return side_effects


def load_additional_side_effects(filename: str, drug: str, existing: Set[str]) -> List[str]:
    """
    Загружает список побочек из JSON и отбирает те, которых ещё нет в графе
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    side_e_list = data.get(drug, {}).get("side_e_parts", [])
    return [s for s in side_e_list if s not in existing]


def add_shifted_nodes(G: Graph, side_effects: List[str], base_coords: Tuple[float, float], shift: float = 50):
    """
    Добавляет побочные эффекты в граф со смещением по X от базовой координаты
    """
    base_x, base_y = base_coords
    for se in side_effects:
        G.add_node(f"{se}(side_e)",
                   label=f"{se}(side_e)",
                   x=str(base_x + shift),
                   y=str(base_y),
                   shape="ellipse")
        

def load_edges_from_csv(path: str, added_side_e: list) -> List[Tuple[str, str]]:
    """
    Загружает список рёбер из CSV-файла (столбцы 'source', 'target')
    Учитываются только те возможные рёбра, которые входят в добавленные побочки
    """
    edges = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source = row.get("source")
            target = row.get("target")
            if source and target and target in added_side_e:
                edges.append((source.strip(), target.strip()))
    return edges


def add_csv_edges_to_graph(G: Graph, edge_list: List[Tuple[str, str]]):
    """
    Добавляет рёбра между узлами, если оба существуют в графе
    """
    node_list = [node.get_label_text_list()[0] for node_id, node in G.nodes.items()]

    for source, target in edge_list:
        if source in node_list and target in node_list:
            G.add_edge( source,
                        target,
                        color = "#FF0000")
        # else:
        #     print(f"Skipped edge: '{source}' -> '{target}' (one or both nodes missing)")


def main():
    # Путь к данным
    input_graph_path = "process_yEd_graph\\data\\graph_yEd_raw_1\\drug_geparin_na.graphml"
    side_effect_json = "make_side_effect_dataset\\data\\sef_dataset.json"
    output_graph_dir = "process_yEd_graph\\data\\test_graphs"
    csv_path = "process_yEd_graph\\data\\list_edges.csv"

    # Загрузка графа
    graph = load_graphml_to_graph(input_graph_path)

    # Поиск препаратов
    drug_name, base_coords = find_prepare_nodes(graph)

    # Поиск уже существующих побочек
    existing_effects = find_side_effects(graph)

    # Загрузка новых побочек
    new_side_effects = load_additional_side_effects(side_effect_json, drug_name, existing_effects)

    # Добавление новых узлов
    add_shifted_nodes(graph, new_side_effects, base_coords, shift=200.0)

    # Загрузка и добавление рёбер из CSV, если файл существует
    edge_list = load_edges_from_csv(csv_path, new_side_effects)
    add_csv_edges_to_graph(graph, edge_list)

    # Сохранение графа
    output_path = os.path.join(output_graph_dir, f"test_{drug_name}_add_side_e.graphml")
    graph.write_graph(output_path, pretty_print=True)
    print(f"Graph saved to: {output_path}")

if __name__ == "__main__":
    main()
