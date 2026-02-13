import json
import sys
sys.path.append("")

from visualization_embedding.get_embeddings import get_embeddings
from visualization_embedding.EmbeddingProcessor import EmbeddingProcessor
from visualization_embedding.cluster_strategy import AgglomerativeStrategy
from visualization_embedding.dimensional_reduct import UMAPStrategy



# Укажите путь к вашему JSON-файлу
file_path = "train_synonim_model\\data\\clusters_2025_08_20_400.json"
model_path = 'train_synonim_model\\data\\synonym-model'
save_dir = 'train_synonim_model\\data'

# Загружаем данные из файла
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Список для хранения строк с null
null_strings = []

# Проходим по кластерам
for cluster_key, cluster_value in data.get('clusters', {}).items():
    for label_pair in cluster_value.get('labels', []):
        if len(label_pair) == 2 and label_pair[1] is None:
            null_strings.append(label_pair[0])

clustering_strategy = AgglomerativeStrategy(n_clusters=15)
reduction_strategy = UMAPStrategy(
                                    n_components=2,
                                    n_neighbors=15,         # Баланс между локальностью и глобальностью
                                    min_dist=0.3,           # Меньше слипания кластеров
                                    metric='cosine',
                                    random_state=42         # Воспроизводимость
                                )
embedding_processor = EmbeddingProcessor(
    reduction_strategy=reduction_strategy,
    clustering_strategy=clustering_strategy
)
embeddings = get_embeddings(null_strings, model_path)
clusters = embedding_processor.clustering_strategy.cluster(embeddings)
embedding_processor.save_result(labels=null_strings, clusters=clusters, save_dir=save_dir)

