import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
from itertools import product
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from dimensional_reduct import TSNEStrategy, TSNEExactStrategy, UMAPStrategy, PCAStrategy, DimensionalityReductionStrategy
from cluster_strategy import DBSCANStrategy, KMeansStrategy, AgglomerativeStrategy, ClusteringStrategy, HDBSCANStrategy
from visualization_strategy import MatplotlibVisualization, PlotlyVisualization, VisualizationStrategy

class ClusterSaver:
    @staticmethod
    def generate_filename(  reduction_method: str, 
                            clustering_method: str,
                            n_clusters: int,
                            base_dir: str = "data") -> str:
        """Генерирует имя файла на основе параметров кластеризации"""
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{reduction_method}_{clustering_method}_clusters{n_clusters}_.json"
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, filename)

    @staticmethod
    def save_clustering_results(
        embeddings: np.ndarray,
        labels: List[str],
        clusters: np.ndarray,
        reduction_params: Dict[str, Any],
        clustering_params: Dict[str, Any],
        reduction_method: str,
        clustering_method: str,
        save_dir: str = "results"
    ) -> str:
        """
        Сохраняет результаты кластеризации и возвращает путь к файлу
        
        :return: Путь к сохраненному файлу
        """
        # Генерируем имя файла
        n_clusters = len(np.unique(clusters))
        filename = ClusterSaver.generate_filename(
            reduction_method=reduction_method,
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            base_dir=save_dir
        )
        
        results = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "reduction_method": reduction_method,
                "clustering_method": clustering_method,
                "num_clusters": len(np.unique(clusters)),
                "num_points": len(embeddings),
                "embedding_dim": embeddings.shape[1]
            },
            "parameters": {
                "reduction": reduction_params,
                "clustering": clustering_params
            },
            "clusters": {
                f"cluster_{int(cluster_num)}": {
                    "count": int(np.sum(clusters == cluster_num)),
                    "labels": [
                        label
                        for label, cluster in zip(labels, clusters)
                        if cluster == cluster_num
                    ],
                }
                for cluster_num in np.unique(clusters)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        return filename


# 4. Класс для работы с эмбеддингами
class EmbeddingVisualizer:
    def __init__(self, 
                 reduction_strategy: 'DimensionalityReductionStrategy',
                 clustering_strategy: 'ClusteringStrategy',
                 visualization_strategy: 'VisualizationStrategy',
                #  primary_reduction_strategy: 'DimensionalityReductionStrategy' = None
                 ):
        """
        Оптимизированный визуализатор эмбеддингов с поддержкой d_mas векторов
        
        Параметры:
            reduction_strategy: Стратегия уменьшения размерности
            clustering_strategy: Стратегия кластеризации
            visualization_strategy: Стратегия визуализации
            # primary_reduction_strategy: Опциональная первичная стратегия уменьшения размерности
        """
        # self.primary_reduction_strategy = primary_reduction_strategy
        self.reduction_strategy = reduction_strategy
        self.clustering_strategy = clustering_strategy
        self.visualization_strategy = visualization_strategy

    def load_data(self, file_path: str) -> Dict[str, Any]:
        """Загрузка данных из JSON файла с обработкой ошибок"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Ошибка загрузки данных: {str(e)}")

    def _validate_inputs(self, embeddings: np.ndarray, section_ids: List[int], num_sections: int):
        """Валидация входных данных для d_mas"""
        if len(embeddings) != len(section_ids):
            raise ValueError("Количество эмбеддингов и section_ids должно совпадать")
        if num_sections <= 0:
            raise ValueError("num_sections должно быть положительным числом")
        if not all(0 <= sid < num_sections for sid in section_ids):
            raise ValueError(f"Все section_ids должны быть в диапазоне [0, {num_sections})")

    def add_one_hot(
            self,
            embeddings: np.ndarray,
            section_list: List[str],
            active_value: float = 6.0,
            inactive_value: float = 0.0,
            normalize: bool = True
        ) -> np.ndarray:
        """
        Оптимизированное добавление one-hot векторов секций
        
        Параметры:
            embeddings: Исходные эмбеддинги (n_samples, n_features)
            section_list: Список секций в соответсвии с эмбеддингом
            num_sections: Общее количество секций
            active_value: Значение активной секции
            inactive_value: Значение неактивных секций
            normalize: Нормализовать ли данные
        
        Возвращает:
            Расширенные эмбеддинги (n_samples, n_features + num_sections)
        """

        # Создаем one-hot векторы
        _, sections = np.unique(section_list, return_inverse=True)  # sections содержит индексы 0..N-1
        one_hot = np.full((len(embeddings), sections.max()+1), inactive_value, dtype=np.float32)
        one_hot[np.arange(len(embeddings)), sections] = active_value
        
        # Объединяем
        result = np.concatenate([embeddings, one_hot], axis=1)

        # section_ids = [idx for idx in range(len(section_list))]
        # num_sections = max(section_ids) + 1

        # self._validate_inputs(embeddings, section_ids, num_sections)
        
        # # Создание one-hot векторов с использованием advanced indexing
        # d_mas_vectors = np.zeros((len(embeddings), num_sections))
        # d_mas_vectors[np.arange(len(embeddings)), section_ids] = active_value
        
        # # Объединение массивов без копирования исходных данных
        # extended_embeddings = np.concatenate(
        #     [embeddings, d_mas_vectors],
        #     axis=1,
        #     dtype=np.float32  # Используем float32 для экономии памяти
        # )
        
        # if normalize:
        #     # Оптимизированная нормализация за один проход
        #     scaler = StandardScaler()
        #     extended_embeddings = scaler.fit_transform(extended_embeddings)
        
        return result

    def load_data(self, data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        Создание списков с данными
        
        Параметры:
            data: Словарь с данными
            add_d_mas: Добавлять ли d_mas векторы
        
        Возвращает:
            Кортеж (эмбеддинги, список меток)
        """
        # Векторизованное извлечение данных
        word_list = list(data.keys())
        embeddings = np.array([item["embedding"] for item in data.values()])
        section_list = np.array([item["section"] for item in data.values()])
               
        return embeddings, word_list, section_list

    def process_embedding(
            self, 
            json_file: dict, 
            title: str = "Embedding Visualization",
            save_results: bool = True,
            visualize: bool = True,
            save_dir: str = "data"
        ) -> Optional[str]:
        """
        Оптимизированный пайплайн визуализации
        
        Параметры:
            json_file: Входные данные
            title: Заголовок визуализации
            save_results: Сохранять результаты
            save_dir: Директория для сохранения
        
        Возвращает:
            Путь к сохранённому файлу или None
        """
        # Обработка данных
        embeddings, labels, section_list = self.load_data(json_file)
        print("self.reduction_strategy:", self.reduction_strategy)
        emb_reduce = self.reduction_strategy.reduce_dimensions(embeddings)
        emb_reduce_one_hot = self.add_one_hot(embeddings=emb_reduce, section_list=section_list)
        cluster_reduce_one_hot = self.clustering_strategy.cluster(emb_reduce_one_hot)

        # Для визуализации
        vis_emb = self.reduction_strategy.reduce_dimensions(emb_reduce_one_hot)
        
        if visualize:
            figure = self.visualization_strategy.visualize(vis_emb, labels, cluster_reduce_one_hot, title)
            self.visualization_strategy.save_visualize(
                                                    reduction_method=self.reduction_strategy.__class__.__name__,
                                                    clustering_method=self.clustering_strategy.__class__.__name__, 
                                                    clusters=cluster_reduce_one_hot,
                                                    fig = figure
                                                )

        # Метрики кластеризации
        # print(f"original emb silhouette score {silhouette_score(embeddings, clusters)}")
        # print(f"primary reduction emb silhouette score {silhouette_score(primary_reduce_embeddings, clusters)}")
        # if self.primary_reduction_strategy:
        #     print(f"secondary reduction emb silhouette score {silhouette_score(secondary_reduce_embeddings, clusters)}")
        
        if save_results:
            return ClusterSaver.save_clustering_results(
                embeddings=emb_reduce_one_hot,
                labels=labels,
                clusters=cluster_reduce_one_hot,
                reduction_params=self.reduction_strategy.__dict__,
                clustering_params=self.clustering_strategy.__dict__,
                reduction_method=self.reduction_strategy.__class__.__name__,
                clustering_method=self.clustering_strategy.__class__.__name__,
                save_dir=save_dir
            )
        return None
    
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
        visualizer: EmbeddingVisualizer,
        json_dict: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        n_jobs: int = -1
    ) -> Tuple[List[Dict[str, Any]], int]:
    """
    Параллельный перебор комбинаций гиперпараметров
    
    Параметры:
        n_jobs: количество рабочих процессов (-1 для использования всех ядер)
    """
    # Загрузка данных один раз
    original_emb, labels, section_list = visualizer.load_data(json_dict)
    
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
        visualizer = EmbeddingVisualizer(
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
        
        # Запуск перебора параметров
        results, emb_dim = grid_search_embeddings_parallel(visualizer, json_dict, param_grid, n_jobs=-1)
        # results, emb_dim = grid_search_embeddings(visualizer, json_dict, param_grid)
        
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
    

# 4. Пример использования
# if __name__ == "__main__":
#     # Конфигурация
#     # EMB_SIDE_E_JSON = "visualization_embedding\\data\\filled_embedding_blank_nodes.json"
#     # EMB_SIDE_E_JSON = "visualization_embedding\\data\\embedding_blank_nodes2_med_embeddings_model_rubert-tiny.json"
#     EMB_SIDE_E_JSON = "visualization_embedding\\data\\embedding_blank_nodes2_med_embeddings_model_all_MiniLM_L6_v2.json"

#     SAVE_DIR = "visualization_embedding\\data\\clustered"

#     with open(EMB_SIDE_E_JSON, "r", encoding = "utf-8") as file:
#         json_dict = json.load(file)["side_effects"]

#     # Выбор стратегий
#     primary_reduction_strategy = TSNEExactStrategy(n_components=50,  random_state=42)
#     reduction_strategy = TSNEStrategy(n_components=2, random_state=42)
#     # reduction_strategy = PCAStrategy(n_components=2)
#     clustering_strategy = DBSCANStrategy(eps=5, min_samples=3)    
#     # clustering_strategy = KMeansStrategy(n_clusters=104)

#     # clustering_strategy = AgglomerativeStrategy(n_clusters=250)    
#     visualization_strategy = PlotlyVisualization()                  
    
#     # Создание и использование визуализатора
#     visualizer = EmbeddingVisualizer(
#         primary_reduction_strategy=primary_reduction_strategy,
#         reduction_strategy=reduction_strategy,
#         clustering_strategy=clustering_strategy,
#         visualization_strategy=visualization_strategy
#     )

#     result_file = visualizer.visualize_embeddings(
#         json_file=json_dict,
#         title="Drug Clustering",
#         save_results=True,
#         save_dir=SAVE_DIR
#     )
#     # visualizer.visualize_embeddings(EMB_SIDE_E_JSON, "Drug Side Effects Embeddings with Clustering")