import networkx as nx
import re
import json
import os
import uuid
import sys
sys.path.append("")

from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer

# dir_processed = "data\\graph_data_yed_processed"
# dir_save = "data\\graph_data_polina"

dir_processed = "data\\graph_data_yEd_test"
dir_save = "data\\graph_data_yEd_test"

custom_morph = EnhancedMorphAnalyzer()

def extract_label_and_tag(G, node):

    node_label = G.nodes[node].get('label', node)

    match = re.search(r"^(.*)\s*\(([^)]+)\)$", node_label)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return node, None


for i, filename in enumerate(os.listdir(dir_processed)):
    if filename.endswith(".graphml"):

        G = nx.read_graphml(f"{dir_processed}\\{filename}")

        json_filename = filename.split(".")[0]+'.json'
        main_prepare = [] 

        # Словарь для сопоставления исходного узла с его новым uuid
        mapping = {}

        G_for_json = nx.DiGraph()

        # Обход всех узлов для создания новых узлов с uuid
        for node in G.nodes():
            # Генерация нового uuid для узла
            new_id = str(uuid.uuid4())
            mapping[node] = new_id

            # Извлечение исходного ярлыка и тега (при необходимости можно сохранить их для отображения)
            node_name, node_tag = extract_label_and_tag(G, node)

            # Лемматизация
            node_name = custom_morph.lemmatize_string(node_name)
            
            # Добавление узла с новым uuid в граф
            G_for_json.add_node(new_id, name=node_name, label=node_tag, weight=5)

            if any(extract_label_and_tag(G, p)[1] == 'group' for p in G.predecessors(node)) and node_tag == 'prepare':
                # print(G.nodes[node].get("name", ""))
                main_prepare.append(new_id)

        # Добавление рёбер от узлов main_prepare к узлам с тегом 'side_e', у которых нет предков
        for node in G.nodes():
            node_uuid = mapping[node]
            _, node_tag = extract_label_and_tag(G, node)
            parents = list(G.predecessors(node))
            if not parents and node_tag == 'side_e':
                for prepare in main_prepare:
                    print(prepare, node_uuid, node_tag, parents == [])
                    G_for_json.add_edge(prepare, node_uuid)

        # Добавление рёбер между узлами согласно исходным рёбрам, используя новые uuid
        for target, source in G.edges():
            target_uuid = mapping[target]
            source_uuid = mapping[source]
            G_for_json.add_edge(target_uuid, source_uuid)

        
        data = nx.node_link_data(G_for_json, edges="links")  # Преобразуем граф в формат, совместимый с JSON

        data["name"] = [G_for_json.nodes[prepare].get("name", "") for prepare in main_prepare]

        with open(f"{dir_save}\\{json_filename}", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


# Путь к директории, которую нужно архивировать
# source_dir = '/путь/к/директории'

# # Имя архива (без расширения)
# archive_name = 'drug_json_graphs'

# # Если архив уже существует, удаляем его (для формата zip)
# zip_path = archive_name + '.zip'
# if os.path.exists(zip_path):
#     os.remove(zip_path)

# # Создание архива
# shutil.make_archive(archive_name, 'zip', dir_save)