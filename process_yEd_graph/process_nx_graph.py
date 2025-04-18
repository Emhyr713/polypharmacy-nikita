import uuid
import matplotlib.pyplot as plt
import networkx as nx
from io import StringIO
import re
import sys
sys.path.append("")

from utils import yedLib

from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer
custom_morph = EnhancedMorphAnalyzer()

class ProcessNxGraph():

    __TAGS_LIST = ['absorbtion', 'action', 'excretion', 'group',
                   'prot_link', 'mechanism', 'metabol', 'hormone',
                   'side_e', 'prepare', 'distribution']

    def __init__(self, morph_analyzer, side_e_dict = {}, flag_debag = False):
        self.morph_analyzer = morph_analyzer
        self.flag_debag = flag_debag
        self.map_side_e_dict = {side_e:side_104 for side_104, side_e in side_e_dict}
        self.nodes_set = set()
        self.edges_set = set()

    @classmethod
    def get_tags_list(cls):
        """
        Геттер списка корректных тегов
        """
        return cls.__TAGS_LIST
    
    def split_by_bracket(self, text):
        """
        Отделение строки со скобкой на 2 строки
        """
        match = re.search(r"^(.*)\s*\(([^)]+)\)$", text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, None
    
    def print_message(self, message, type_message = "DEBAG"):
        if type_message != "DEBAG" or self.flag_debag:
            print(f"     {type_message}:{message}")

    def extract_label_and_tag(self, G, node):
        """
        Отделение от названия узла сущность и её тег
        """
        node_label = G.nodes[node].get('label', node)
        return self.split_by_bracket(node_label)
    
    def find_main_prepare(self, G, include_tag = False):
        """
        Поиск ЛС в графе
        """
        main_prepare = []

        # Поиск ЛС в графе
        for node in G.nodes():
            parents = G.predecessors(node)
            parent_tags = [self.extract_label_and_tag(G, parent)[1] for parent in parents]
            node_label, node_tag = self.extract_label_and_tag(G, node)
            if parents and any(tag == 'group' for tag in parent_tags) and node_tag == 'prepare':
                if include_tag:
                    main_prepare.append(node)
                else:
                    main_prepare.append(node_label)
        
        return main_prepare
    

    def concat_hanging_act_mech(self, G):
        """
        Этап 1: Конкатенация висячих узлов с тегами 'action'/'mechanism',
        имеющих потомка с тегом 'noun', и зацепление с главными веществами.
        """

        def find_hanging_act_mech(G):
            """
            Функция для поиска висячих узлов с тегами 'action' или 'mechanism',
            у которых нет входящих рёбер и хотя бы один потомок имеет тег 'noun'
            """

            return [
                    node for node in G.nodes()
                    if self.extract_label_and_tag(G, node)[1] in ('action', 'mechanism')
                    and not list(G.predecessors(node))
                    and any(self.extract_label_and_tag(G, child)[1] == 'noun' for child in G.successors(node))
                ]


        nodes_to_remove = set() # Список на удаление

        # Этап 1: Конкатенация висячих action/mechanism, имеющих потомка с тегом 'noun'
        hanging_act_mech = find_hanging_act_mech(G)

        while hanging_act_mech != []:
            for parent in hanging_act_mech:
                # Находим всех потомков с тегом 'noun' у родительского узла
                noun_children = [
                    child for child in list(G.successors(parent))
                    if self.extract_label_and_tag(G, child)[1] == 'noun'
                ]
                if any(self.extract_label_and_tag(G, child)[1] != 'noun' for child in list(G.successors(parent))):
                    self.print_message(f"{self.extract_label_and_tag(G, parent)} has been not noun in child", "WARNING")
                        
                if not noun_children:
                    raise RuntimeWarning("Висячая вершина не может иметь в потомках не noun")
            
                label_parent, tag_parent = self.extract_label_and_tag(G, parent)
                for child in noun_children:            
                    label_child, tag_child = self.extract_label_and_tag(G, child)
                    new_node = f"{label_parent} {label_child}({tag_parent})"
                    G.add_node(new_node, label=new_node)

                    for prechild in list(G.predecessors(child)):
                        if prechild != parent:
                            G.add_edge(prechild, new_node)
                    
                    # Добавление рёбер от нового узла к потомкам узла-потомка
                    for grandchild in list(G.successors(child)):
                        G.add_edge(new_node, grandchild)

                    self.print_message(f"new_node:{new_node},"
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
            if self.extract_label_and_tag(G, node)[1] in ('action', 'mechanism') and G.in_degree(node) == 0
        ]
        main_prepares = self.find_main_prepare(G, include_tag=True)
        for node in hanging_act_mech:
            for prepare in main_prepares:
                G.add_edge(prepare, node)

        return G
    
    def remove_remaining_noun(self, G):
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
            node_name, node_tag = self.extract_label_and_tag(G, node)
            children = list(G.successors(node))

            if node_tag is None:
                self.print_message(f"Node:{node}, Node_name:{node_name}, Node_tag:{node_tag} is NONE", "WARNING")
            
            # print(f"Processing node: {node_name} (Tag: {node_tag})")
            
            for child in children:
                # child_label = G.nodes[child].get('label',node)
                child_name, child_tag = self.extract_label_and_tag(G, child)
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
            child_tags = [self.extract_label_and_tag(G, child)[1] for child in children]
            if children and all(tag == 'noun' for tag in child_tags):
                nodes_to_remove.add(node)

        # Удаление узлов после завершения обработки
        for remove_node in nodes_to_remove:
            if remove_node in G:
                # print(f"Node {G.nodes[remove_node].get('label',remove_node)} has been removed.")
                G.remove_node(remove_node)
        
        return G
    

    def collect_nodes_edges(self, G):
        """
        Сохнанить уникальные узлы и связи
        """

        main_prepare = self.find_main_prepare(G)
            
        # По всем узлам графа
        for node in G.nodes():
            label, tag = self.extract_label_and_tag(G, node)
            self.nodes_set.add((label, tag))

            if tag not in ProcessNxGraph.get_tags_list():
                self.print_message(f"label:{label}, tag:{tag}, prepare:{main_prepare}", "INVALID TAG")

            if tag != "side_e" and not list(G.predecessors(node)) and not list(G.successors(node)):
                self.print_message(f"label:{label}, tag:{tag}, prepare:{main_prepare}", "INVALID EDGE")

            # # Связывание несвязанных побочных эффектов
            # if tag == "side_e" and not list(G.predecessors(node)):
            #     for prepare in main_prepare:
            #         self.edges_set.add((prepare, label))

        # По всем рёбрам
        for target, source in G.edges():
            label_t, tag_t = self.extract_label_and_tag(G, target)
            label_s, tag_s = self.extract_label_and_tag(G, source)

            if tag_t in ProcessNxGraph.get_tags_list() and tag_s in ProcessNxGraph.get_tags_list():
                self.edges_set.add((f"{label_t}({tag_t})", f"{label_s}({tag_s})"))

        return self.nodes_set, self.edges_set
    

    def link_isolated_side_e(self, G):
        """
        Прикрепление изолированных вершин побочек
        """

        G_new = G.copy()

        # Поиск вершин ЛС, к которым прикрепятся изолированные side_e
        main_prepare = self.find_main_prepare(G, include_tag=True)

        # Добавление рёбер от узлов 'prepare' к 'side_e', у которых нет предков
        for node in G.nodes():
            _, node_tag = self.extract_label_and_tag(G, node)
            parents = list(G.predecessors(node))
            if not parents and node_tag == 'side_e':
                for prepare in main_prepare:
                    G_new.add_edge(prepare, node)

        return G_new


    def replacing_side_e_with_dict(self, G):
        """
        Замена побочек на побочки из словаря
        (Проверить)
        """

        G_new = G.copy()

        # Добавление рёбер от узлов 'prepare' к 'side_e', у которых нет предков
        for node in G.nodes():
            node_label, node_tag = self.extract_label_and_tag(G, node)
            if node_tag == 'side_e':

                new_name = self.side_e_dict.get(node_label, node_label)

                # new_name = # ПОИСК ПОДХОДЯЩЕЙ ЗАМЕНЫ на побочку

                # Добавить новый узел
                G_new.add_node(new_name, label=f"{node_label}({node_tag})")

                # Перенести входящие рёбра (u → A)
                for u, _ in list(G.in_edges(node)):
                    G_new.add_edge(u, new_name)

                # Перенести исходящие рёбра (A → v)
                for _, v in list(G.out_edges(node)):
                    G_new.add_edge(new_name, v)

            # Удалить старый узел
            G.remove_node(node)

        return G_new


    def merge_graphs(self, graphs):
        """
        Объединяет несколько графов, сохраняя и объединяя мета-данные узлов.
        
        Args:
            graphs (list): Список графов для объединения.
        
        Returns:
            networkx.Graph: Новый граф с объединенными узлами, ребрами и мета-данными.
        """
        if not graphs:
            print("На вход не поданы графы")
            return nx.Graph()

        # Выбираем тип результирующего графа
        directed = any(g.is_directed() for g in graphs)
        M = nx.DiGraph() if directed else nx.Graph()

        # Добавляем все узлы по их label
        for G in graphs:
            for n, data in G.nodes(data=True):
                lab = data.get('label', str(n))
                if lab not in M:
                    M.add_node(lab, label=lab)

        # Добавляем все рёбра, переведя концы в их label
        for G in graphs:
            for u, v in G.edges():
                lu = G.nodes[u].get('label', str(u))
                lv = G.nodes[v].get('label', str(v))
                if not M.has_edge(lu, lv):
                    M.add_edge(lu, lv)

        return M
        
    @staticmethod
    def print_nodes(G):
        """
        Печать узлов
        """
        for node in G.nodes:
            print(node)

    @staticmethod
    def plot_graph(G):
        """
        Вывод графа на экран
        """
        pos = nx.shell_layout(G)  
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=6, edge_color="gray")
        plt.show()

    def graph2json(self, G):
        """
        Обрабатывает граф в формате JSON.
        (проверить)
        """
        
        main_prepare = []           # Список узлов с тегом 'prepare', связанных с 'group'
        mapping = {}                # Сопоставление старых узлов с новыми UUID
        G_for_json = nx.DiGraph()
        
        # Обход всех узлов и создание новых узлов с UUID
        for node in G.nodes():
            new_id = str(uuid.uuid4())
            mapping[node] = new_id
            
            node_name, node_tag = self.extract_label_and_tag(G, node)
            if self.morph_analyzer:
                node_name = self.morph_analyzer.lemmatize_string(node_name)
            
            G_for_json.add_node(new_id, name=node_name, label=node_tag, weight=5)
            
            if any(self.extract_label_and_tag(G, p)[1] == 'group' for p in G.predecessors(node)) and node_tag == 'prepare':
                main_prepare.append(new_id)
                
        # Добавление рёбер между узлами, используя новые UUID
        for target, source in G.edges():
            target_uuid = mapping[target]
            source_uuid = mapping[source]
            G_for_json.add_edge(target_uuid, source_uuid)
        
        # Преобразование в JSON-формат
        data = nx.node_link_data(G_for_json, edges="links")
        data["name"] = [G_for_json.nodes[prepare].get("name", "") for prepare in main_prepare]

        return data
    
    def yEd2graph(self, file_path):
        """
        Загружает .graphml, лемматизирует текст узлов (label),
        если они в формате 'текст(тег)',
        и возвращает готовый граф networkx.
        """

        # Читаем содержимое XML-файла
        with open(file_path, 'r', encoding='utf-8') as f:
            xml_text = f.read()

        # Функция для замены текста внутри <y:NodeLabel>
        def repl(match):
            before, text, after = match.groups()
            label, tag = self.split_by_bracket(text)

            if not label or not tag:
                return match.group(0)  # вернуть как есть, если не подходит под шаблон

            lemmatized = self.morph_analyzer.lemmatize_string(label)
            return f"{before}{lemmatized}({tag}){after}"

        # Заменяем тексты в узлах
        modified_xml = re.sub(
            r'(<y:NodeLabel[^>]*>)(.*?)(</y:NodeLabel>)',
            repl,
            xml_text,
            flags=re.DOTALL
        )

        # Передаём в networkx из памяти
        return nx.read_graphml(StringIO(modified_xml))
    
    def add_side_e_from_dataset(self, G, dataset):
        pass
    

    def graph2yEd(self, G, loaded_yed = False):
        graph_yed = yedLib.Graph()
        # Словарь для хранения идентификаторов узлов
        node_ids = {}
        node_set = set()

        # Добавление узлов
        for i, node in enumerate(G.nodes()):

            if loaded_yed:
                label = G.nodes[node].get("label", str(node))
                node_id = label
            else:
                name = G.nodes[node].get("name", str(node))
                label = G.nodes[node].get("label", str(node))
                node_id = f"{name}({label})"

            node_ids[node] = node_id
            if node_id not in node_set:
                graph_yed.add_node(node_id, shape="ellipse")
                node_set.add(node_id)
            else:
                print(f"Узел {node_id} уже существует, добавление пропущено.")


        # Добавление рёбер
        for i, (source, target, edge_data) in enumerate(G.edges(data=True)):
            graph_yed.add_edge(node_ids[source], node_ids[target], arrowhead="standard")

        return graph_yed.get_graph()
    
    def save_nxGraph_to_yEd(self, G, path):
        xml_str = self.graph2yEd(G, loaded_yed=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml_str)



if __name__ == "__main__":

    file_lizinopril = "process_yEd_graph\\data\\graph_yEd_raw_1\\drug_lizinopril.graphml"
    file_spironolacton = "process_yEd_graph\\data\\graph_yEd_raw_1\\drug_spironolakton.graphml"
    

    process_graph = ProcessNxGraph(custom_morph)

    G_lizinopril = process_graph.yEd2graph(file_lizinopril)
    G_spironolacton = process_graph.yEd2graph(file_spironolacton)

    # Этап 1
    G_lizinopril = process_graph.concat_hanging_act_mech(G_lizinopril)
    G_spironolacton = process_graph.concat_hanging_act_mech(G_spironolacton)
    # Этап 2
    G_lizinopril = process_graph.remove_remaining_noun(G_lizinopril)
    G_spironolacton = process_graph.remove_remaining_noun(G_spironolacton)
    # Этап 3 
    G_lizinopril = process_graph.link_isolated_side_e(G_lizinopril)
    G_spironolacton = process_graph.link_isolated_side_e(G_spironolacton)

    print("Main prepare G_lizinopril:", process_graph.find_main_prepare(G_lizinopril))
    print("Main prepare G_spironolacton:", process_graph.find_main_prepare(G_spironolacton))

    DIR_TEST = "process_yEd_graph\\data\\test_graphs"

    process_graph.save_nxGraph_to_yEd(G_lizinopril, f"{DIR_TEST}\\lizinopril.graphml")
    process_graph.save_nxGraph_to_yEd(G_spironolacton, f"{DIR_TEST}\\spironolacton.graphml")

    merged_graph = process_graph.merge_graphs([G_lizinopril, G_spironolacton])
    process_graph.save_nxGraph_to_yEd(merged_graph, f"{DIR_TEST}\\merged_lizinopril_spironolacton.graphml")

    G_1 = process_graph.yEd2graph(DIR_TEST+"\\test_merge_1.graphml")
    G_2 = process_graph.yEd2graph(DIR_TEST+"\\test_merge_2.graphml")

    merged_graph = process_graph.merge_graphs([G_1, G_2])
    process_graph.save_nxGraph_to_yEd(merged_graph, f"{DIR_TEST}\\merged_1_2.graphml")

    # graph_test = process_graph.yEd2graph(DIR_TEST+"\\test_3_before.graphml")
    # process_graph.concat_hanging_act_mech(graph_test)
    # process_graph.remove_remaining_noun(graph_test)
    # process_graph.link_isolated_side_e(graph_test)
    # process_graph.save_yEd(graph_test, f"{DIR_TEST}\\test_3_after.graphml")

