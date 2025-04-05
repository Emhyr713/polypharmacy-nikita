import json
import os
import uuid
import networkx as nx
import re
from typing import Any, Tuple

from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer

# dir_processed = "data\\graph_data_yed_processed"
# dir_save = "data\\graph_data_polina"

dir_processed = "data\\graph_data_yEd_test"
dir_save = "data\\graph_data_yEd_test"


class yEd2json:
    """
    Класс для обработки GraphML-файлов, преобразования их в JSON-совместимый формат
    и выполнения предобработки узлов и связей.
    """
    
    def __init__(self, morph_analyzer: Any = None) -> None:
        """
        Инициализация класса.
        
        :param input_path: Путь к файлу или директории с GraphML файлами.
        :param output_dir: Директория для сохранения JSON-файлов (если не указано, используется рабочая папка).
        :param morph_analyzer: Инструмент для лемматизации текста (если передан).
        """
        # self.input_path = input_path
        # self.output_dir = output_dir if output_dir else os.getcwd()
        self.morph_analyzer = morph_analyzer

    @staticmethod
    def extract_label_and_tag(G: nx.Graph, node: str) -> Tuple[str, str]:
        """
        Извлекает текстовый ярлык и тег из узла.
        
        :param G: Граф networkx.
        :param node: Идентификатор узла.
        :return: Кортеж (ярлык, тег).
        """
        node_label = G.nodes[node].get('label', node)
        match = re.search(r"^(.*)\s*\(([^)]+)\)$", node_label)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return node, None

    def process(self, input_path: str, output_dir: str = None) -> None:
        """
        Обрабатывает либо один файл, либо все файлы в директории.
        """
        # print("input_path:", input_path, os.path.isfile(input_path), input_path.endswith(".graphml"))
        # output_dir = output_dir or self.output_dir
        if os.path.isdir(input_path):
            self.process_graphml_files(input_path, output_dir)
        elif input_path.endswith(".graphml"):
            return self.process_single_graphml(input_path)
        else:
            raise ValueError("Указанный путь не является файлом GraphML или директорией.")

    def process_graphml_files(self, input_path:str, output_dir:str = "") -> None:
        """
        Обрабатывает все GraphML-файлы в указанной директории и сохраняет их в JSON-формате.
        """
        for filename in os.listdir(input_path):
            if filename.endswith(".graphml"):
                json_filename = filename.replace(".graphml", ".json")
                data = self.process_single_graphml(filename)
                self.save_graph_json(json_filename, data, output_dir)

    def process_single_graphml(self, filename: str) -> None:
        """
        Обрабатывает один GraphML-файл и сохраняет его в формате JSON.
        
        :param filename: Имя файла GraphML.
        """
        G = nx.read_graphml(filename)
        
        main_prepare = []  # Список узлов с тегом 'prepare', связанных с 'group'
        mapping = {}  # Сопоставление старых узлов с новыми UUID
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
        
        # Добавление рёбер от узлов 'prepare' к 'side_e', у которых нет предков
        for node in G.nodes():
            node_uuid = mapping[node]
            _, node_tag = self.extract_label_and_tag(G, node)
            parents = list(G.predecessors(node))
            if not parents and node_tag == 'side_e':
                for prepare in main_prepare:
                    G_for_json.add_edge(prepare, node_uuid)
        
        # Добавление рёбер между узлами, используя новые UUID
        for target, source in G.edges():
            target_uuid = mapping[target]
            source_uuid = mapping[source]
            G_for_json.add_edge(target_uuid, source_uuid)
        
        # Преобразование в JSON-формат
        data = nx.node_link_data(G_for_json, edges="links")
        data["name"] = [G_for_json.nodes[prepare].get("name", "") for prepare in main_prepare]

        return data


    def save_graph_json(self, data: str, json_filename: str, output_dir: str = "") -> None:
        """
        Сохраняет graph в json.
        
        :param filename: Имя файла .json.
        :param data: Данные о графе.
        :param output_dir: Директория сохранения.
        """
        # Сохранение JSON-файла
        output_path = os.path.join(output_dir, json_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

            
if __name__ == "__main__":
    # dir_processed = "path_to_graphml_files"
    # dir_save = "path_to_save_json"
    morph_analyzer = EnhancedMorphAnalyzer()
    
    processor = yEd2json(dir_processed, dir_save, morph_analyzer)
    processor.process_graphml_files()
