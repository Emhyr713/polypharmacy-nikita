import json
import networkx as nx

import sys
sys.path.append("")

from convertors.graph2yEd import graph2yEd

from convertors.yed2json import yEd2json
from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer

custom_morph = EnhancedMorphAnalyzer()

class GraphMerger:
    """
    Класс для объединения двух графов, представленных в формате JSON.
    Узлы объединяются по ключу (name, label), а связи корректируются с учётом объединения узлов.
    Также предоставляется метод для преобразования объединённого графа в объект networkx.
    """
    
    def __init__(self, graph1, graph2) -> None:
        """
        Инициализация класса с двумя графами.
        
        :param graph1: Первый граф в формате JSON.
        :param graph2: Второй граф в формате JSON.
        """
        self.graph1 = graph1
        self.graph2 = graph2
        self.merged_graph = {
            "directed": graph1["directed"] or graph2["directed"],
            "multigraph": graph1["multigraph"] or graph2["multigraph"],
            "graph": {},
            "nodes": [],
            "links": []
        }
        self.node_map  = {}     # ключ (name, label) -> список ID узлов
        self.id2key  = {}       # id узла -> ключ (name, label)

    def _merge_nodes(self) -> None:
        """
        Объединяет узлы из обоих графов по ключу (name, label). Если узел с таким ключом уже встречался,
        его ID добавляются в список. При этом все узлы добавляются в объединённый граф.
        """
        for node in self.graph1["nodes"] + self.graph2["nodes"]:
            key = (node["name"], node["label"])
            if key not in self.node_map:
                self.node_map[key] = []  # Инициализируем список ID для данного ключа
            self.node_map[key].append(node["id"])
            self.id2key[node["id"]] = key
            self.merged_graph["nodes"].append(node)

    def _merge_links(self) -> None:
        """
        Объединяет связи (links) из обоих графов, заменяя source и target на первый ID для соответствующего узла.
        Дублирующиеся связи не добавляются.
        """
        link_set = set()
        merged_links = []
        
        for link in self.graph1["links"] + self.graph2["links"]:
            # Получаем ключ для исходного и целевого узлов по их ID
            source_key = self.id2key.get(link["source"])
            target_key = self.id2key.get(link["target"])
            if source_key is None or target_key is None:
                continue  # Пропускаем связь, если не найден соответствующий узел
            
            # Берем первый ID из списка для каждого узла
            new_source = self.node_map[source_key][0]
            new_target = self.node_map[target_key][0]
                            
            if (new_source, new_target) not in link_set:
                link_set.add((new_source, new_target))
                merged_links.append({"source": new_source, "target": new_target})
        
        self.merged_graph["links"] = merged_links

    def merge(self):
        """
        Объединяет узлы и связи двух графов.
        
        :return: Объединенный граф в формате JSON.
        """
        self._merge_nodes()
        self._merge_links()
        return self.merged_graph
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Преобразует объединённый граф в объект networkx.
        
        :return: Граф в формате networkx (DiGraph, если граф направленный, иначе Graph).
        """
        # Создаем граф в зависимости от направленности объединенного графа
        G = nx.DiGraph() if self.merged_graph["directed"] else nx.Graph()
        
        for node in self.merged_graph["nodes"]:
            G.add_node(
                node["id"],
                name=node["name"],
                label=node["label"],
                weight=node.get("weight", 1)
            )
        
        for link in self.merged_graph["links"]:
            G.add_edge(link["source"], link["target"])
    
        return G

    def save_to_file(self, filename: str) -> None:
        """
        Сохраняет объединённый граф в файл в формате JSON.
        
        :param filename: Имя файла для сохранения.
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.merged_graph, f, ensure_ascii=False, indent=4)

def load_graph(filename: str):
    """
    Загружает граф из JSON или GraphML файла.
    
    :param filename: Имя файла с графом.
    :return: Граф в формате JSON (словарь Python).
    """
    if filename.endswith(".json"):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    elif filename.endswith(".graphml"):
        processor_yEd2json = yEd2json(morph_analyzer=custom_morph)
        return processor_yEd2json.process(filename)  # Преобразование в JSON-совместимый формат
    else:
        raise ValueError("Поддерживаются только файлы JSON и GraphML.")

# Использование
if __name__ == "__main__":
    a, b = 2,3
    graph1 = load_graph(f"data\\graph_yEd_processed_1\\drug_fozinopril.graphml")
    graph2 = load_graph(f"data\\graph_yEd_processed_1\\drug_ramipril.graphml")

    merger = GraphMerger(graph1, graph2)
    merged_graph = merger.merge()
    nx_merged_graph = merger.to_networkx()
    xml_str = graph2yEd(nx_merged_graph)

    # Записываем в файл
    output_path = f"process_yEd_graph\\data\\test_graphs\\merged_graph_fozinopril_ramipril"
    merger.save_to_file(f"{output_path}.json")
    with open(f"{output_path}.graphml", "w", encoding="utf-8") as f:
        f.write(xml_str)

