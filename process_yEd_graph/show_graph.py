import json
import networkx as nx
import matplotlib.pyplot as plt

# Загрузка JSON-файла
with open('process_yEd_graph\\data\\test_bayes_graph_evgeny.json',
          'r',
          encoding='utf-8') as f:
    data = json.load(f)

# Создание графа
G = nx.node_link_graph(data)

# Сопоставление id и name для подписей
labels = {node['id']: node['name'] for node in data['nodes']}

# Отрисовка
pos = nx.spring_layout(G)
nx.draw(
    G, 
    pos, 
    labels=labels,  # Используем имена вместо id
    with_labels=True,
    node_color='lightgreen',
    node_size=1000,
    edge_color='gray',
    font_size=12,
    font_weight='bold'
)

plt.title("Граф с именами узлов")
plt.show()