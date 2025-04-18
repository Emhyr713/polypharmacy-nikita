import os
import json
import csv
import re
import pandas as pd
import networkx as nx
import sys
sys.path.append("")

import xml.etree.ElementTree as ET

from utils.yedLib import Graph, Node
from process_nx_graph import ProcessNxGraph
from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer
custom_morph = EnhancedMorphAnalyzer()

# Регистрируем префиксы, чтобы при сохранении они остались
ET.register_namespace("", "http://graphml.graphdrawing.org/xmlns")
ET.register_namespace("y", "http://www.yworks.com/xml/graphml")

ns = {
    "g": "http://graphml.graphdrawing.org/xmlns",
    "y": "http://www.yworks.com/xml/graphml"
}

def split_by_bracket(text):
    """
    Отделение строки со скобкой на 2 строки
    """
    match = re.search(r"^(.*)\s*\(([^)]+)\)$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


# Парсинг graphml графа
def load_graphml_to_graph(path: str) -> Graph:
    # Открываем файл в UTF-8
    with open(path, 'r', encoding='utf-8') as f:
        tree = ET.parse(f)
    root = tree.getroot()

    graph_el = root.find("g:graph", ns)
    g = Graph(directed=graph_el.get("edgedefault"),
              graph_id=graph_el.get("id"))

    for node_el in graph_el.findall("g:node", ns):
        node_id = node_el.get("id")

        # 1) найти первый data, в котором есть ShapeNode
        data_el = node_el.find("g:data[y:ShapeNode]", ns)
        if data_el is None:
            # возможно изолированный узел без визуалки
            g.add_node(node_id)
            continue

        shape_el = data_el.find("y:ShapeNode", ns)
        if shape_el is None:
            g.add_node(node_id)
            continue

        # 2) Geometry
        geom = shape_el.find("y:Geometry", ns)
        if geom is None:
            x = y = 0.0
            w = h = None
        else:
            x = float(geom.get("x", "0"))
            y = float(geom.get("y", "0"))
            w = geom.get("width")
            h = geom.get("height")

        # 3) Label
        label_el = shape_el.find("y:NodeLabel", ns)
        label = label_el.text if label_el is not None else node_id

        # 4) Добавляем узел с атрибутами в ваш Graph
        g.add_node(node_id,
                   label=label,
                   x=str(x), y=str(y),
                   shape = "ellipse",
                   width=w, height=h)

    # Рёбра
    for edge_el in graph_el.findall("g:edge", ns):
        src = edge_el.get("source")
        tgt = edge_el.get("target")
        g.add_edge(src, tgt)

    return g

def add_shifted_from_prepare(G, side_e_list, coords, shift = 50):

    for new_node in side_e_list:
        G.add_node(f"{new_node}(side_e)",
                    label=f"{new_node}(side_e)",
                    x=str(coords[0]+shift),
                    y=str(coords[1]),
                    shape="ellipse")

def find_prepare(G):
    result = []

    for node_id, node in G.nodes.items():
        for label in node.get_label_text_list():
            text = label.lower()
            if "prepare" in text:
                label, tag = split_by_bracket(text)
                
                result.append([label, float(node.geom["x"]), float(node.geom["y"])])

    # Собираем все label'ы, x и y
    labels = [label for label, _, _ in result]
    xs = [x for _, x, _ in result]
    ys = [y for _, _, y in result]

    # Формируем drug_name
    drug_name = '+'.join(sorted(labels))

    # Самая правая (max x) и самая верхняя (min y) координаты
    rightmost_x = max(xs)
    topmost_y = min(ys)

    print(drug_name, rightmost_x, topmost_y)
    return drug_name, (rightmost_x, topmost_y)

def find_side_e(G):
    side_list = set()

    for node_id, node in G.nodes.items():
        for label in node.get_label_text_list():
            text = label.lower()
            if "side_e" in text:
                label, tag = split_by_bracket(text)
                
                side_list.add(label)

    return side_list

def load_side_e(filename, drug, existing_side_e):
    with open(filename, "r", encoding="utf-8") as f:
        side_e_list = json.load(f)[drug]["side_e_parts"]

    not_exist_side_e = [side_e for side_e in side_e_list if side_e not in list(existing_side_e)]
    return not_exist_side_e
            
if __name__ == "__main__":
    # 1. загрузка
    G = load_graphml_to_graph("process_yEd_graph\\data\\graph_yEd_raw_1\\drug_linagliptin_empalogiphlozin.graphml")

    # 2. Поиск целевых препаратов в графе и их координат
    prepare, coords = find_prepare(G)

    # 3. Поиск побочек из текущего графа
    existing_side_e = find_side_e(G)

    # 4. Загрузка побочных эффектов
    side_e_list = load_side_e("make_side_effect_dataset\\data\\sef_dataset.json", prepare, existing_side_e)

    # 5. добавляем сдвинутые узлы
    add_shifted_from_prepare(G, side_e_list, coords, shift=200.0)

    # 6. сохраняем
    G.write_graph(f"process_yEd_graph\\data\\test_graphs\\test_{prepare}_add_side_e.graphml", pretty_print=True)
