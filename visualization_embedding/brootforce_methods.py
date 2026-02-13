import json
from typing import List, Dict, Any, Optional, Tuple

from itertools import product
from tqdm import tqdm

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from dimensional_reduct import TSNEStrategy, TSNEExactStrategy, UMAPStrategy, PCAStrategy, DimensionalityReductionStrategy
from cluster_strategy import DBSCANStrategy, KMeansStrategy, AgglomerativeStrategy, ClusteringStrategy, HDBSCANStrategy
from visualization_strategy import MatplotlibVisualization, PlotlyVisualization, VisualizationStrategy

import visualization_embedding.EmbeddingProcessor as EmbeddingProcessor

def evaluate_combination(visualizer, original_emb, section_list, labels, reduction_strategy, clustering_strategy):
    """Вычисляет метрики для одной комбинации параметров"""
    # Сценарий 1: Добавление вектора, снижение размерности, кластеризация
    emb_one_hot = visualizer.add_one_hot(embeddings=original_emb, section_list=section_list)
    emb_one_hot_reduce = reduction_strategy.reduce_dimensions(emb_one_hot)
    cluster_emb_one_hot_reduce = clustering_strategy.cluster(emb_one_hot_reduce)

    # Сценарий 2: Снижение размерности
    emb_reduce = reduction_strategy.reduce_dimensions(original_emb)
    # Сценарий 2.1: Кластеризация
    cluster_emb_reduce = clustering_strategy.cluster(emb_reduce)
    # Сценарий 2.2: Добавление вектора, кластеризация
    emb_reduce_one_hot = visualizer.add_one_hot(embeddings=emb_reduce, section_list=section_list)
    cluster_reduce_one_hot = clustering_strategy.cluster(emb_reduce_one_hot)

    # Выбор уникальных кластеров
    unique_clusters_one_hot_reduce = np.unique(cluster_emb_one_hot_reduce)
    unique_clusters_reduce = np.unique(cluster_emb_reduce)
    unique_clusters_reduce_one_hot = np.unique(cluster_reduce_one_hot)
    
    # Подсчёт количества кластеров
    n_clusters_one_hot_reduce = len(unique_clusters_one_hot_reduce[unique_clusters_one_hot_reduce != -1])
    n_clusters_reduce = len(unique_clusters_reduce[unique_clusters_reduce != -1])
    n_clusters_reduce_one_hot = len(unique_clusters_reduce_one_hot[unique_clusters_reduce_one_hot != -1])

    # Подсчёт по стратегии 1
    score_one_hot_reduce = (silhouette_score(emb_one_hot_reduce, cluster_emb_one_hot_reduce) 
                           if n_clusters_one_hot_reduce > 1 else float('nan'))
    dbi_score_one_hot_reduce = (davies_bouldin_score(emb_one_hot_reduce, cluster_emb_one_hot_reduce) 
                              if n_clusters_one_hot_reduce > 1 else float('nan'))

    # Подсчёт по стратегии 2.1
    score_reduce = (silhouette_score(emb_reduce, cluster_emb_reduce) 
                   if n_clusters_reduce > 1 else float('nan'))
    dbi_score_reduce = (davies_bouldin_score(emb_reduce, cluster_emb_reduce) 
                      if n_clusters_reduce > 1 else float('nan'))

    # Подсчёт по стратегии 2.2
    score_reduce_one_hot = (silhouette_score(emb_reduce_one_hot, cluster_reduce_one_hot) 
                          if n_clusters_reduce_one_hot > 1 else float('nan'))
    dbi_score_reduce_one_hot = (davies_bouldin_score(emb_reduce_one_hot, cluster_reduce_one_hot) 
                               if n_clusters_reduce_one_hot > 1 else float('nan'))

    return {
        'algorithm': {
            'reduction': str(reduction_strategy.__class__.__name__),
            'clustering': str(clustering_strategy.__class__.__name__)
        },
        'hyperparameters': {
            'reduction': reduction_strategy.get_params(),
            'clustering': clustering_strategy.get_params()
        },
        'n_clusters_one_hot_reduce': n_clusters_one_hot_reduce,
        'n_clusters_reduce': n_clusters_reduce,
        'n_clusters_reduce_one_hot': n_clusters_reduce_one_hot,
        'score_one_hot_reduce': score_one_hot_reduce,
        'score_reduce': score_reduce,
        'score_reduce_one_hot': score_reduce_one_hot,
        'dbi_score_one_hot_reduce': dbi_score_one_hot_reduce,
        'dbi_score_reduce': dbi_score_reduce,
        'dbi_score_reduce_one_hot': dbi_score_reduce_one_hot,
    }

def grid_search_embeddings_parallel(
        visualizer: EmbeddingProcessor,
        original_emb, labels, section_list,
        param_grid: Dict[str, List[Any]],
        n_jobs: int = -1
    ) -> Tuple[List[Dict[str, Any]], int]:
    """
    Параллельный перебор комбинаций гиперпараметров
    
    Параметры:
        n_jobs: количество рабочих процессов (-1 для использования всех ядер)
    """
    # Загрузка данных один раз
    # original_emb, labels, section_list = visualizer.load_data(json_dict)
    
    # Генерация всех комбинаций параметров
    combinations = list(product(param_grid['reduction'], param_grid['clustering']))
    
    # Параллельное выполнение
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_combination)(
            visualizer, original_emb, section_list, labels, reduction, clustering
        )
        for reduction, clustering in tqdm(combinations, desc="Processing combinations")
    )
    
    return results, len(original_emb[0])    

def plot_sorted_score_histograms(namefile, data_list, top_n=7):
    """
    Отрисовка гистограмм с метриками
    """

    # Разворачиваем данные: создаём отдельные записи для каждой стратегии
    flattened_data = []
    for data in data_list:
        for strategy in ['one_hot_reduce', 'reduce', 'reduce_one_hot']:
            flattened_data.append({
                'algorithm': data['algorithm'],
                'hyperparameters': data['hyperparameters'],
                'strategy': strategy,
                'silhouette_score': data[f'score_{strategy}'],
                'dbi_score': data[f'dbi_score_{strategy}'],
                'n_clusters': data[f'n_clusters_{strategy}']
            })

    # Сортируем по Silhouette Score (по убыванию) и берём топ-N
    sorted_data = sorted(flattened_data, key=lambda x: x['silhouette_score'], reverse=True)[:top_n]
    
    # Настройка графика
    fig, ax = plt.subplots(figsize=(20, 7))
    plt.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.75)
    
    bar_width = 0.35
    group_spacing = 1.0  # Расстояние между группами
    
    # Цвета
    silhouette_color = '#1f77b4'  # синий
    dbi_color = '#d62728'        # красный
    
    # Позиции столбцов
    x = np.arange(len(sorted_data)) * group_spacing
    
    # Отрисовка
    for i, data in enumerate(sorted_data):
        x_pos = x[i]
        
        # Данные
        s_score = data['silhouette_score']
        dbi_score = data['dbi_score']
        n_clusters = data['n_clusters']
        strategy = data['strategy']
        
        reduction_name = data['algorithm']['reduction']
        clustering_name = data['algorithm']['clustering']
        reduction_params = "\n".join(f"{k}: {v}" for k, v in data['hyperparameters']['reduction'].items())
        clustering_params = "\n".join(f"{k}: {v}" for k, v in data['hyperparameters']['clustering'].items())
        
        # Столбцы
        ax.bar(x_pos - bar_width/2, s_score, bar_width, color=silhouette_color, alpha=0.7, label='Silhouette' if i == 0 else "")
        ax.bar(x_pos + bar_width/2, dbi_score, bar_width, color=dbi_color, alpha=0.7, label='DBI' if i == 0 else "")
        
        # Подписи значений
        ax.text(x_pos - bar_width/2, s_score, f'{s_score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(x_pos + bar_width/2, dbi_score, f'{dbi_score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Подпись под столбцами
        strategy_label = {
            'one_hot_reduce': 'OneHot → Reduce',
            'reduce': 'Reduce',
            'reduce_one_hot': 'Reduce → OneHot'
        }[strategy]
        
        config_text = (
            f"{reduction_name}\n+\n{clustering_name}\n\n"
            f"{strategy_label}\n\n"
            f"Clusters: {n_clusters}\n\n"
            f"Reduction params:\n{reduction_params}\n\n"
            f"Clustering params:\n{clustering_params}"
        )
        
        ax.text(x_pos, -0.05,
               config_text, ha='center', va='top', fontsize=12)
    
    # Легенда, оси, заголовок
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylabel('Score Value', fontsize=12)
    ax.set_title(f'Первые {len(sorted_data)} cтратегий отсортированные по silhouette score', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'{namefile}.png', dpi=1000, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Загрузка данных
    NAMES = ["all_MiniLM_L6_v2",
             "Clinical_ModernBERT",
             "distiluse_base_multilingual",
             "paraphrase_multilingual",
             "rubert-tiny",
             "rubert-tiny2"
             ]
    
    for NAME in NAMES:
        # NAME = "rubert-tiny2"
        N_TRY = "3"
        EMB_SIDE_E_JSON = f"visualization_embedding/data/embedding_blank_nodes2_med_embeddings_model_{NAME}.json"
        with open(EMB_SIDE_E_JSON, "r", encoding="utf-8") as file:
            json_dict = json.load(file)["side_effects"]
        
        # Создание базового визуализатора
        visualization_strategy = PlotlyVisualization()
        visualizer = EmbeddingProcessor(
            primary_reduction_strategy=None,
            reduction_strategy=UMAPStrategy(n_components=2),
            clustering_strategy=HDBSCANStrategy(),
            visualization_strategy=visualization_strategy
        )
        
        # Сетка параметров для перебора
        param_grid = {
            'reduction': [
                UMAPStrategy(n_components=50, n_neighbors=10, min_dist=0.1, metric='cosine'),
                UMAPStrategy(n_components=100, n_neighbors=10, min_dist=0.1, metric='cosine'),
                UMAPStrategy(n_components=2, n_neighbors=10, min_dist=0.1, metric='cosine'),
                TSNEStrategy(n_components=2, random_state=42),
                TSNEStrategy(n_components=3, random_state=42),
                TSNEExactStrategy(n_components=50, random_state=42),
                TSNEExactStrategy(n_components=100, random_state=42),
                PCAStrategy(n_components=2),
                PCAStrategy(n_components=50),
                PCAStrategy(n_components=100),
                PCAStrategy(n_components=0.95)
            ],
            'clustering': [
                # HDBSCANStrategy(min_cluster_size=2, min_samples=2, cluster_selection_epsilon=0.002, cluster_selection_method='eom',metric='cosine'),
                # HDBSCANStrategy(min_cluster_size=2, min_samples=2, cluster_selection_epsilon=0.002, cluster_selection_method='eom',metric='euclidean'),
                
                AgglomerativeStrategy(n_clusters=100),
                AgglomerativeStrategy(n_clusters=150),
                AgglomerativeStrategy(n_clusters=200),
                KMeansStrategy(n_clusters=100),
                KMeansStrategy(n_clusters=150),
                KMeansStrategy(n_clusters=200),

                # DBSCANStrategy(eps=0.7, min_samples=3),
                # DBSCANStrategy(eps=1.1, min_samples=3),
                # DBSCANStrategy(eps=2.5, min_samples=3),
                # DBSCANStrategy(eps=5, min_samples=3)
            ]
        }

        # Получение данных из словаря
        embeddings, labels, section_list = visualizer.load_data_from_json(json_dict)
        # Запуск перебора параметров
        results, emb_dim = grid_search_embeddings_parallel(visualizer, embeddings,
                                                           labels, section_list,
                                                           param_grid, n_jobs=-1)
        
        # Вывод лучших результатов
        print("\nTop combinations by final silhouette score:")
        sorted_results = sorted(
            [r for r in results if not np.isnan(r['score_one_hot_reduce'])],
            key=lambda x: x['score_one_hot_reduce'],
            reverse=True
        )

        filename = f"visualization_embedding\\data\\cluster_result_{NAME}_{N_TRY}"
        plot_sorted_score_histograms(filename, results)
        
        with open(f"{filename}.txt", "w", encoding = 'utf-8') as f:
            f.write(f"model: {NAME}\n")
            f.write(f"embedding dimentions: {emb_dim}\n")

            for i, result in enumerate(sorted_results, 1):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"RESULT #{i}\n")
                f.write(f"{'=' * 80}\n\n")
                
                # Основные метрики
                f.write(f"{'METRICS':-^40}\n")
                f.write(f"Number of cluster one_hot_reduce: {result['n_clusters_one_hot_reduce']}\n")
                f.write(f"Number of cluster reduce: {result['n_clusters_reduce']}\n")
                f.write(f"Number of cluster reduce_one_hot: {result['n_clusters_reduce_one_hot']}\n\n")

                f.write(f"score_one_hot_reduce: {result['score_one_hot_reduce']:.4f}\n")
                f.write(f"score_reduce: {result['score_reduce']:.4f}\n")
                f.write(f"score_reduce_one_hot: {result['score_reduce_one_hot']:.4f}\n\n")

                f.write(f"dbi_score_one_hot_reduce: {result['dbi_score_one_hot_reduce']:.4f}\n")
                f.write(f"dbi_score_reduce: {result['dbi_score_reduce']:.4f}\n")
                f.write(f"dbi_score_reduce_one_hot: {result['dbi_score_reduce_one_hot']:.4f}\n\n")
                
                # Алгоритмы
                f.write(f"{'ALGORITHMS':-^40}\n")
                # f.write(f"Primary reduction: {result['algorithm']['primary_reduction'] or 'None'}\n")
                f.write(f"Final reduction: {result['algorithm']['reduction']}\n")
                f.write(f"Clustering: {result['algorithm']['clustering']}\n\n")

                # f.write(f"hot_one_mass: {result['hot_one_mass']}\n")
                
                # Гиперпараметры
                f.write(f"{'HYPERPARAMETERS':-^40}\n")
                
                # if result['hyperparameters']['primary_reduction']:
                #     f.write("Primary reduction:\n")
                #     for k, v in result['hyperparameters']['primary_reduction'].items():
                #         f.write(f"  {k}: {v}\n")
                
                f.write("\nFinal reduction:\n")
                for k, v in result['hyperparameters']['reduction'].items():
                    f.write(f"  {k}: {v}\n")
                
                f.write("\nClustering:\n")
                for k, v in result['hyperparameters']['clustering'].items():
                    f.write(f"  {k}: {v}\n")
                
                f.write("\n")