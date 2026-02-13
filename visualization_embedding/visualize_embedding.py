import json

import dimensional_reduct #import TSNEStrategy, TSNEExactStrategy, UMAPStrategy, PCAStrategy, DimensionalityReductionStrategy
import cluster_strategy #import DBSCANStrategy, KMeansStrategy, AgglomerativeStrategy, ClusteringStrategy, HDBSCANStrategy
import visualization_strategy #import MatplotlibVisualization, PlotlyVisualization, VisualizationStrategy
from visualization_embedding.EmbeddingProcessor import EmbeddingProcessor

# 4. Пример использования
if __name__ == "__main__":
    # Конфигурация
    # EMB_SIDE_E_JSON = "visualization_embedding\\data\\filled_embedding_blank_nodes.json"
    # EMB_SIDE_E_JSON = "visualization_embedding\\data\\embedding_blank_nodes2_med_embeddings_model_rubert-tiny.json"
    # EMB_SIDE_E_JSON = "visualization_embedding\\data\\embedding_blank_nodes2_med_embeddings_model_all_MiniLM_L6_v2.json"

    EMB_SIDE_E_JSON = "visualization_embedding\\data\\embedding_blank_nodes2_med_embeddings_model_distiluse_base_multilingual.json"

    SAVE_DIR = "visualization_embedding\\data\\clustered"

    with open(EMB_SIDE_E_JSON, "r", encoding = "utf-8") as file:
        json_dict = json.load(file)["side_effects"]

    # Выбор стратегий
    reduction_strategy = dimensional_reduct.UMAPStrategy(n_components=2,
                                                         n_neighbors=10,
                                                         min_dist=0.1,
                                                         metric='cosine')
    clustering_strategy = cluster_strategy.AgglomerativeStrategy(n_clusters=200)
    visualization_strategy = visualization_strategy.PlotlyVisualization()                  
    
    # Создание и использование визуализатора
    visualizer = EmbeddingProcessor(
        reduction_strategy=reduction_strategy,
        clustering_strategy=clustering_strategy,
        visualization_strategy=visualization_strategy
    )

    result_file = visualizer.process_embedding(
        json_file=json_dict,
        title="Drug Clustering",
        save_results=True,
        save_dir=SAVE_DIR
    )
    # visualizer.visualize_embeddings(EMB_SIDE_E_JSON, "Drug Side Effects Embeddings with Clustering")