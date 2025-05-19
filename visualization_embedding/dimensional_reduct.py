from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap 

class DimensionalityReductionStrategy(ABC):
    @abstractmethod
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        pass

    def get_params(self):
        pass

class PCAStrategy(DimensionalityReductionStrategy):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(embeddings)
    
    def get_params(self):
        return {
            'n_components': self.n_components
        }
    

class UMAPStrategy(DimensionalityReductionStrategy):
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
        **kwargs
    ):
        """
        Стратегия уменьшения размерности с использованием UMAP
        
        Параметры:
            n_components (int): Конечная размерность (по умолчанию 2)
            n_neighbors (int): Количество соседей для построения графа (по умолчанию 15)
            min_dist (float): Минимальное расстояние между точками (по умолчанию 0.1)
            metric (str): Метрика расстояния ('cosine', 'euclidean' и др.)
            random_state (int): Seed для воспроизводимости
            **kwargs: Дополнительные параметры для UMAP
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.kwargs = kwargs

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Применяет UMAP для уменьшения размерности embeddings
        
        Аргументы:
            embeddings: Входные embedding'ы (размерность [n_samples, n_features])
            
        Возвращает:
            np.ndarray: Уменьшенные embedding'ы [n_samples, n_components]
        """
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            **self.kwargs
        )
        return reducer.fit_transform(embeddings)
    
    def get_params(self):
        return {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric':self.metric
        }

class TSNEStrategy(DimensionalityReductionStrategy):
    def __init__(self, n_components: int = 2, random_state: int = 42):
        if n_components > 3:
            raise ValueError("For barnes_hut method (default), n_components should be 2 or 3. "
                           "For higher dimensions, use method='exact'")
        self.n_components = n_components
        self.random_state = random_state
        
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        tsne = TSNE(n_components=self.n_components, 
                   random_state=self.random_state)
        return tsne.fit_transform(embeddings)
    
    def get_params(self):
        return {
            'n_components': self.n_components,
            'random_state': self.random_state
        }

class TSNEExactStrategy(DimensionalityReductionStrategy):
    """Альтернативная стратегия для n_components > 3"""
    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        tsne = TSNE(n_components=self.n_components,
                   method='exact',
                   random_state=self.random_state)
        return tsne.fit_transform(embeddings)
    
    def get_params(self):
        return {
            'n_components': self.n_components,
            'random_state': self.random_state
        }