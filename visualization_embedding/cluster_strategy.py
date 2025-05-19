from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
import hdbscan

class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        pass

    def get_params(self):
        pass

class KMeansStrategy(ClusteringStrategy):
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return kmeans.fit_predict(embeddings)
    
    def get_params(self):
        return {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }

class DBSCANStrategy(ClusteringStrategy):
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return dbscan.fit_predict(embeddings)
    
    def get_params(self):
        return {
            'eps': self.eps,
            'min_samples': self.min_samples
        }

class AgglomerativeStrategy(ClusteringStrategy):
    def __init__(self, n_clusters: int = 5, linkage: str = 'ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        agg = AgglomerativeClustering(n_clusters=self.n_clusters,
                                      linkage=self.linkage,
                                      )
        return agg.fit_predict(embeddings)
    
    def get_params(self):
        return {
            'n_clusters': self.n_clusters,
            'linkage': self.linkage
        }
    
class HDBSCANStrategy(ClusteringStrategy):
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        cluster_selection_epsilon: float = 0.0,
        alpha: float = 1.0,
        normalize_embeddings: bool = True,  # Новый параметр
        **kwargs
    ):
        """
        Стратегия кластеризации с использованием HDBSCAN
        
        Параметры:
            min_cluster_size (int): Минимальный размер кластера
            min_samples (int): Минимальное количество соседей для core точки
            metric (str): Метрика расстояния ('euclidean', 'cosine' и др.)
            cluster_selection_method (str): Метод выбора кластеров ('eom' или 'leaf')
            cluster_selection_epsilon (float): Параметр для объединения близких кластеров
            alpha (float): Параметр сглаживания для иерархии кластеров
            **kwargs: Дополнительные параметры для HDBSCAN
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.alpha = alpha
        self.normalize_embeddings = normalize_embeddings
        self.kwargs = kwargs

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Применяет HDBSCAN для кластеризации embedding'ов
        
        Аргументы:
            embeddings: Входные embedding'ы (размерность [n_samples, n_features])
            
        Возвращает:
            np.ndarray: Массив меток кластеров [n_samples], где -1 - шум
        """

        if self.metric == 'cosine':
            # Преобразуем cosine similarity в distance матрицу
            embeddings = normalize(embeddings, norm='l2')
            distance_matrix = cosine_distances(embeddings)
            distance_matrix = distance_matrix.astype(np.float64)
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_method=self.cluster_selection_method,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                alpha=self.alpha,
                metric='precomputed',  # Используем предвычисленную матрицу
                # **{k:v for k,v in self.__dict__.items() if k != 'metric' and 'min_cluster_size'}
                
            )
            return clusterer.fit_predict(distance_matrix)
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_method=self.cluster_selection_method,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                alpha=self.alpha,
                **self.kwargs
            )
            return clusterer.fit_predict(embeddings)
    
    def get_params(self):
        return {
            'min_cluster_size': self.min_cluster_size,
            'metric': self.metric,
        }
    
    