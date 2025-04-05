# import json
import csv
import matplotlib.pyplot as plt
import networkx as nx
import os
import re
import sys
sys.path.append("")

import sqlite3

from convertors.graph2yEd import graph2yEd

flag_debag = False

def load_graphml(file_path):
    return nx.read_graphml(file_path)

def extract_label_and_tag(G, node):
    # pattern = r"^(.*?)\(([^)]+)\)\s*$"
    
    node_label = G.nodes[node].get('label', node)

    match = re.search(r"^(.*)\s*\(([^)]+)\)$", node_label)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return node, None

def find_main_prepare(G, include_tag = False):
    main_prepare = []

    # Поиск ЛС в графе
    for node in G.nodes():
        parents = G.predecessors(node)
        parent_tags = [extract_label_and_tag(G, parent)[1] for parent in parents]
        node_label, node_tag = extract_label_and_tag(G, node)
        if parents and any(tag == 'group' for tag in parent_tags) and node_tag == 'prepare':
            if include_tag:
                main_prepare.append(node)
            else:
                main_prepare.append(node_label)
    
    return main_prepare

def print_message(message, type_message = "DEBAG"):
    global flag_debag
    if type_message != "DEBAG" or flag_debag:
        print(f"     {type_message}:{message}")

# Функция для поиска висячих узлов с тегами 'action' или 'mechanism',
# у которых нет входящих рёбер и хотя бы один потомок имеет тег 'noun'
def find_hanging_act_mech(G):
    return [
            node for node in G.nodes()
            if extract_label_and_tag(G, node)[1] in ('action', 'mechanism')
            and not list(G.predecessors(node))
            and any(extract_label_and_tag(G, child)[1] == 'noun' for child in G.successors(node))
        ]


def concat_hanging_act_mech(G):
    """
    Этап 1: Конкатенация висячих узлов с тегами 'action'/'mechanism',
    имеющих потомка с тегом 'noun', и зацепление с главными веществами.
    """

    nodes_to_remove = set() # Список на удаление

    # Этап 1: Конкатенация висячих action/mechanism, имеющих потомка с тегом 'noun'
    hanging_act_mech = find_hanging_act_mech(G)

    while hanging_act_mech != []:
        for parent in hanging_act_mech:
            # Находим всех потомков с тегом 'noun' у родительского узла
            noun_children = [
                child for child in list(G.successors(parent))
                if extract_label_and_tag(G, child)[1] == 'noun'
            ]
            if any(extract_label_and_tag(G, child)[1] != 'noun' for child in list(G.successors(parent))):
                print_message(f"{extract_label_and_tag(G, parent)} has been not noun in child", "WARNING")
                     
            if not noun_children:
                raise RuntimeWarning("Висячая вершина не может иметь в потомках не noun")
        
            label_parent, tag_parent = extract_label_and_tag(G, parent)
            for child in noun_children:            
                label_child, tag_child = extract_label_and_tag(G, child)
                new_node = f"{label_parent} {label_child}({tag_parent})"
                G.add_node(new_node, label=new_node)

                for prechild in list(G.predecessors(child)):
                    if prechild != parent:
                        G.add_edge(prechild, new_node)
                
                # Добавление рёбер от нового узла к потомкам узла-потомка
                for grandchild in list(G.successors(child)):
                    G.add_edge(new_node, grandchild)

                print_message(f"new_node:{new_node},"
                      f"->new_node: {list(G.predecessors(new_node))},"
                      f"new_node->: {list(G.successors(new_node))}")
                
                # Удаляем узел потомка с тегом 'noun'
                nodes_to_remove.add(child)

            # Удаляем родительский узел
            G.remove_node(parent)
        
        # Пересчитываем список висячих action/mechanism узлов,
        # у которых есть потомок с тегом 'noun'
        hanging_act_mech = find_hanging_act_mech(G)

    # Удаление узлов после завершения обработки
    for remove_node in nodes_to_remove:
        if remove_node in G:
            G.remove_node(remove_node)

    # Зацепление с главными веществами для оставшихся висячих узлов с тегами 'action'/'mechanism',
    # у которых нет потомков с тегом 'noun'
    hanging_act_mech = [
        node for node in list(G.nodes())
        if extract_label_and_tag(G, node)[1] in ('action', 'mechanism') and G.in_degree(node) == 0
    ]
    main_prepares = find_main_prepare(G, include_tag=True)
    for node in hanging_act_mech:
        for prepare in main_prepares:
            G.add_edge(prepare, node)

    return G


def remove_remaining_noun(G):
    """
    Этап 2: Обход графа и удаление оставшихся узлов с тегом 'noun'.
    При этом происходит конкатенация родительских и дочерних узлов, если у дочернего тег 'noun'.
    """

    nodes_to_remove = set() # Список на удаление
    queue = list(G.nodes)   # Очередь для обработки узлов (динамически пополняется)
    i = 0                   # Индекс для отладки

    # Пока очередь из улов не кончилась
    while i < len(queue):
        node = queue[i]
        i += 1

        # Пропуск, если узел будет всё равно удалён
        if node in nodes_to_remove:
            continue       

        nodes_to_add = []

        # node_label = G.nodes[node].get('label',node)
        node_name, node_tag = extract_label_and_tag(G, node)
        children = list(G.successors(node))

        if node_tag is None:
            print_message(f"Node:{node}, Node_name:{node_name}, Node_tag:{node_tag} is NONE", "WARNING")
        
        # print(f"Processing node: {node_name} (Tag: {node_tag})")
        
        for child in children:
            # child_label = G.nodes[child].get('label',node)
            child_name, child_tag = extract_label_and_tag(G, child)
            if child_tag == "noun":
                new_node = f"{node_name} {child_name}({node_tag})"
                nodes_to_add.append(new_node)

                # Добавление нового узела и добавление его в очередь
                G.add_node(new_node, label = new_node)
                queue.append(new_node)  

                # "Наследование" связей от текущего обрабатываемого узла
                for parent in G.predecessors(node):
                    G.add_edge(parent, new_node)
                for child_successor in G.successors(child):
                    G.add_edge(new_node, child_successor)

                # Добавление в список удаления
                nodes_to_remove.add(child)  

        # Если все дочерние узлы имеют тег 'noun', то родительский узел подлежит удалению
        child_tags = [extract_label_and_tag(G, child)[1] for child in children]
        if children and all(tag == 'noun' for tag in child_tags):
            nodes_to_remove.add(node)

    # Удаление узлов после завершения обработки
    for remove_node in nodes_to_remove:
        if remove_node in G:
            # print(f"Node {G.nodes[remove_node].get('label',remove_node)} has been removed.")
            G.remove_node(remove_node)
    
    return G


def collect_nodes_edges(G, nodes_set, edges_set):

    main_prepare = find_main_prepare(G)
        
    # По всем узлам графа
    for node in G.nodes():
        label, tag = extract_label_and_tag(G, node)
        nodes_set.add((label, tag))

        if tag not in ('absorbtion', 'action', 'excretion', 'group', 'prot_link', 'mechanism', 'metabol', 'hormone', 'side_e', 'prepare'):
            print_message(f"label:{label}, tag:{tag}, prepare:{main_prepare}", "INVALID TAG")

        if tag != "side_e" and not list(G.predecessors(node)) and not list(G.successors(node)):
            print_message(f"label:{label}, tag:{tag}, prepare:{main_prepare}", "INVALID EDGE")

        # Связывание несвязанных побочных эффектов
        if tag == "side_e" and not list(G.predecessors(node)):
            for prepare in main_prepare:
                edges_set.add((prepare, label))

    # По всем рёбрам
    for target, source in G.edges():
        label_t, tag_t = extract_label_and_tag(G, target)
        label_s, tag_s = extract_label_and_tag(G, source)
        edges_set.add((label_t, label_s))

    
    return nodes_set, edges_set


def save_in_sqltable(table_name, nodes_set, edges_set):
    # Создание и подключение к базе данных (в случае её отсутствия будет создана)
    conn = sqlite3.connect(table_name)
    cursor = conn.cursor()

    # Удаление таблиц, если они существуют
    cursor.execute("DROP TABLE IF EXISTS nodes")
    cursor.execute("DROP TABLE IF EXISTS edges")
    
    # Создание таблицы для узлов
    cursor.execute('''
    CREATE TABLE nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        tag TEXT
    )
    ''')

    # Создание таблицы для рёбер
    cursor.execute('''
    CREATE TABLE edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER,
        target_id INTEGER,
        FOREIGN KEY (source_id) REFERENCES nodes(id),
        FOREIGN KEY (target_id) REFERENCES nodes(id)
    )
    ''')

    # Функция для добавления узлов в таблицу
    def add_nodes_to_db(nodes_set):
        for label, tag in nodes_set:
            cursor.execute("INSERT INTO nodes (label, tag) VALUES (?, ?)", (label, tag))

    # Функция для добавления рёбер в таблицу
    def add_edges_to_db(edges_set):
        for source, target in edges_set:
            # Находим ids для source и target
            cursor.execute("SELECT id FROM nodes WHERE label = ?", (source,))
            source_id = cursor.fetchone()
            cursor.execute("SELECT id FROM nodes WHERE label = ?", (target,))
            target_id = cursor.fetchone()

            if source_id and target_id:
                cursor.execute("INSERT INTO edges (source_id, target_id) VALUES (?, ?)", (source_id[0], target_id[0]))

    # Заполнение базы данных
    add_nodes_to_db(nodes_set)
    add_edges_to_db(edges_set)

    # Сохраняем изменения и закрываем соединение
    conn.commit()

    # Закрытие соединения
    conn.close()


# Печать узлов
def print_nodes(G):
    for node in G.nodes:
        print(node)

# Вывод на экран
def plot_graph(G):
    pos = nx.shell_layout(G)  
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=6, edge_color="gray")
    plt.show()

# Тест 1
def test_graph_1():
    G = nx.DiGraph()
    G.add_node("уменьшение образование угольный кислота(action)")
    G.add_node("снижение реабсорбция(action)")
    G.add_node("эпителий каналец(noun)")
    G.add_node("бикарбонат(noun)")
    G.add_node("ион натрий(noun)")
    G.add_node("повышать экскреции гидрокабонат(action)")
    G.add_node("электролитный нарушение(side_e)")
    G.add_node("увеличение выделение вода(action)")

    G.add_edge("уменьшение образование угольный кислота(action)", "снижение реабсорбция(action)")
    G.add_edge("снижение реабсорбция(action)", "эпителий каналец(noun)")
    G.add_edge("снижение реабсорбция(action)", "тест 123(action)")
    G.add_edge("эпителий каналец(noun)", "бикарбонат(noun)")
    G.add_edge("эпителий каналец(noun)", "ион натрий(noun)")
    G.add_edge("бикарбонат(noun)", "электролитный нарушение(side_e)")
    G.add_edge("ион натрий(noun)", "электролитный нарушение(side_e)")
    G.add_edge("бикарбонат(noun)", "повышать экскреции гидрокабонат(action)")
    G.add_edge("ион натрий(noun)", "увеличение выделение вода(action)")
    return G

# Тест 2
def test_graph_2():
    G = nx.DiGraph()
    G.add_node("mech(mechanism)")
    G.add_node("уменьшение образование угольный кислота(mechanism)")
    G.add_node("эпителий каналец(noun)")
    G.add_node("повышать экскреции гидрокабонат(action)")
    G.add_node("увеличение выделение вода(action)")

    G.add_edge("mech(mechanism)", "уменьшение образование угольный кислота(mechanism)")
    G.add_edge("уменьшение образование угольный кислота(mechanism)", "эпителий каналец(noun)")
    G.add_edge("эпителий каналец(noun)", "повышать экскреции гидрокабонат(action)")
    G.add_edge("уменьшение образование угольный кислота(mechanism)", "увеличение выделение вода(action)")
    return G

# Тест 3
def test_graph_3():
    G = nx.DiGraph()
    G.add_node("mech1(mechanism)")
    G.add_node("mech2(mechanism)")
    G.add_node("noun1(noun)")
    G.add_node("noun2(noun)")
    G.add_node("noun3(noun)")
    G.add_node("action1(action)")

    G.add_edge("mech1(mechanism)", "mech2(mechanism)")
    G.add_edge("mech1(mechanism)", "noun1(noun)")
    G.add_edge("mech1(mechanism)", "noun2(noun)")
    G.add_edge("mech1(mechanism)", "noun3(noun)")
    G.add_edge("noun1(noun)", "noun2(noun)")
    G.add_edge("noun3(noun)", "noun2(noun)")
    G.add_edge("noun2(noun)", "action1(action)")

    return G


if __name__ == "__main__":

    dir_graph_edit = "data\\graph_data_yEd_edit"
    dir_graph_processed = "data\\graph_data_yed_processed"
    csv_nodes = "list_nodes.csv"
    csv_edges = "list_edges.csv"

    nodes_set = set()
    edges_set = set()

    for i, filename in enumerate(os.listdir(dir_graph_edit)):
        if filename.endswith(".graphml"):

            print("filename opened:", filename)

            # Загрузка графа
            G = load_graphml(f"{dir_graph_edit}\\{filename}")
            # Этап 1
            G = concat_hanging_act_mech(G)
            # Этап 2
            G = remove_remaining_noun(G)
            # Подсчёт вершин и рёбер
            nodes_set, edges_set = collect_nodes_edges(G, nodes_set, edges_set)

            xml_str = graph2yEd(G, loaded_yed=True)
            with open(f"{dir_graph_processed}\\{filename}", "w", encoding="utf-8") as f:
                f.write(xml_str)


    # Сохранение в csv для узлов
    header_nodes = ["label", "tag"]
    with open(f"{dir_graph_processed}\\{csv_nodes}", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")   # Указываем delimiter при создании writer
        writer.writerow(header_nodes)           # Записываем заголовки
        writer.writerows(nodes_set)             # Записываем кортежи

    # Сохранение в csv для рёбер
    header_edges = ["source", "target"]  # Пример для рёбер (можно изменить по вашим данным)
    with open(f"{dir_graph_processed}\\{csv_edges}", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")   # Указываем delimiter при создании writer
        writer.writerow(header_edges)           # Записываем заголовки
        writer.writerows(edges_set)             # Записываем кортежи

    print_message(f"nodes_set:{len(nodes_set)}, edges_set:{len(edges_set)}")

    save_in_sqltable(f"{dir_graph_processed}\\graph.db", nodes_set, edges_set)