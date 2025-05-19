import json
import os
import sys
sys.path.append("")

from process_yEd_graph.process_nx_graph import ProcessNxGraph


TEXT_DATASET_FILENAME = "extract_text_from_instructions\\data\\extracted_data_all.json"
DIR_SIDE_E = "process_yEd_graph\\data\\graph_yEd_processed_"

process_graph = ProcessNxGraph()

text_corpus = ""
result = {}

# Извлечение текста
with open(TEXT_DATASET_FILENAME, "r", encoding="utf-8") as file:
    data = json.load(file)
for item in data:
    text_corpus+=item["text"]

# Извлечение узлов
for i in range(2):
    current_dir = f"{DIR_SIDE_E}{i+1}"
    for filename in os.listdir(current_dir):
        if not filename.endswith(".graphml"):
            continue

        # Загрузка графа из файла
        G_yEd = process_graph.load_graphml(f"{current_dir}\\{filename}")

        def extract_nodes_to_dict(graph, tag_type):
            nodes = process_graph.find_node_by_tag(graph, find_tag=tag_type)
            label_tag_pairs = [process_graph.extract_label_and_tag(graph, node) for node in nodes]
            return {label: {"section": tag, "embedding": None} for label, tag in label_tag_pairs}

        # Собираем данные по action и mechanism
        result.update(extract_nodes_to_dict(G_yEd, "action"))
        result.update(extract_nodes_to_dict(G_yEd, "mechanism"))
        result.update(extract_nodes_to_dict(G_yEd, "metabol"))
        result.update(extract_nodes_to_dict(G_yEd, "absorbtion"))
        result.update(extract_nodes_to_dict(G_yEd, "excretion"))
        result.update(extract_nodes_to_dict(G_yEd, "prot_link"))

FILE_SAVE = "visualization_embedding\\data\\embedding_blank_nodes.json"
with open(FILE_SAVE, 'w', encoding='utf-8') as f:
    json.dump({"text":text_corpus,"words":result}, f, indent=4, ensure_ascii=False)
        

   