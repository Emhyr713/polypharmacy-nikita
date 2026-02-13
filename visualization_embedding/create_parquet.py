import json
import pandas as pd

import dimensional_reduct
import cluster_strategy

from get_embeddings import get_embeddings
from EmbeddingProcessor import EmbeddingProcessor
from cluster_strategy import SimilarityThresholdClustering

if __name__ == "__main__":

    # Загрузка терминов
    # TERMS_FILENAME = "visualization_embedding\\data\\terms_list.json"
    # TERMS_FILENAME = "visualization_embedding\\data\\terms_list_verified_folder.json"

    # TERMS_FILENAME = "visualization_embedding\\data\\cautions_dataset.json"
    # TERMS_FILENAME = "visualization_embedding\\data\\contraindications_dataset.json"
    TERMS_FILENAME = "visualization_embedding\\data\\side_effects_dataset.json"


    with open(TERMS_FILENAME, "r", encoding = "utf-8") as file:
        terms_list = json.load(file)

    def load_terms_section(data):
        return [item[0] for item in data], [item[1] for item in data]

    labels, sections = load_terms_section(terms_list)

    # Получение эмбеддингов
    model_path = 'train_synonim_model\\data\\synonym-model_3'
    embeddings = get_embeddings(labels, model_path)
    print("Эмбеддинги получены")

    # # Инициализация обработчика эмбеддингов и выбор стратегий
    reduction_strategy = dimensional_reduct.UMAPStrategy(
                                                        n_components=2,
                                                        n_neighbors=15,         # Баланс между локальностью и глобальностью
                                                        min_dist=0.15,          # Меньше слипания кластеров
                                                        metric='cosine',
                                                        random_state=42         # Воспроизводимость
                                                    )
    clustering_strategy = cluster_strategy.AgglomerativeStrategy(n_clusters=1400)
    # SimilarityThresholdClustering(threshold=0.97)
    embedding_processor = EmbeddingProcessor(
        reduction_strategy=reduction_strategy,
        clustering_strategy=clustering_strategy
    )

    clusters = embedding_processor.clustering(embeddings)


    # Получение 2d кластеризированных эмбеддингов 
    # vis_emb, clusters = embedding_processor.process_embedding(embeddings, sections, return_mode='vis_emb')
    print("Кластеризация и уменьшение размерности выполнены")

    # # Сохранение в Parquet (бинарный формат, эффективный для векторов)
    # PARQUET_FILENAME = "visualization_embedding\\data\\embeddings_3.parquet"
    # df = pd.DataFrame({
    #     "text": labels,
    #     "embedding": embeddings.tolist(),
    #     "clusters": clusters,
    #     'projection_x': vis_emb[:, 0],
    #     'projection_y': vis_emb[:, 1]
    # })
    # df.to_parquet(PARQUET_FILENAME)
    # print(f"dataframe сохранён в {PARQUET_FILENAME}")

    # # Сохранение в json
    dir = "visualization_embedding\\data\\clustered"
    embedding_processor.save_result(labels=labels, clusters=clusters, save_dir=dir)
    