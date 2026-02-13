from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os
import sys
sys.path.append("")

import numpy as np

from visualization_embedding import cluster_strategy 
from visualization_embedding import dimensional_reduct
from visualization_embedding import visualization_strategy

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
        # embeddings: np.ndarray,
        labels: List[str],
        clusters: np.ndarray,
        save_dir: str,

        reduction_params: Dict[str, Any],
        clustering_params: Dict[str, Any],
        reduction_method: str,
        clustering_method: str,
        
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
                "num_points": len(labels),
                # "embedding_dim": embeddings.shape[1]
            },
            "parameters": {
                "reduction": reduction_params,
                "clustering": clustering_params
            },
            "clusters": {
                f"cluster_{int(cluster_num)}": {
                    "count": int(np.sum(clusters == cluster_num)),
                    "labels": [
                        (label, None)
                        for label, cluster in zip(labels, clusters)
                        if cluster == cluster_num
                    ],
                }
                for cluster_num in np.unique(clusters)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Сохранено в {filename}")
        return filename
    

# Класс для работы с эмбеддингами
class EmbeddingProcessor:
    def __init__(self, 
                 reduction_strategy: 'dimensional_reduct' = None,
                 clustering_strategy: 'cluster_strategy' = None,
                 ):
        """
        Оптимизированный визуализатор эмбеддингов с поддержкой d_mas векторов
        
        Параметры:
            reduction_strategy: Стратегия уменьшения размерности
            clustering_strategy: Стратегия кластеризации
        """
        self.reduction_strategy = reduction_strategy
        self.clustering_strategy = clustering_strategy

    def load_data_from_json(self, file_path: str) -> Dict[str, Any]:
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
            active_value: float = 4.0,
            inactive_value: float = 0.0
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
        
        return result

    def load_data_from_json(self, data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
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
    
    def visualize_embeddings(self, vis_emb,
                             labels, cluster_reduce_one_hot,
                             visualization_strategy,
                             title: str = "Embedding Visualization"
                             ):
        figure = self.visualization_strategy.visualize(vis_emb, labels, cluster_reduce_one_hot, title)
        visualization_strategy.save_visualize(
                                                reduction_method=self.reduction_strategy.__class__.__name__,
                                                clustering_method=self.clustering_strategy.__class__.__name__, 
                                                clusters=cluster_reduce_one_hot,
                                                fig = figure
                                            )
        
    def save_result(self, labels, clusters, save_dir):
        return(
        ClusterSaver.save_clustering_results(
            labels=labels,
            clusters=clusters,
            save_dir=save_dir,

            reduction_params=self.reduction_strategy.__dict__,
            clustering_params=self.clustering_strategy.__dict__,
            reduction_method=self.reduction_strategy.__class__.__name__,
            clustering_method=self.clustering_strategy.__class__.__name__
        ))
    
    def clustering(self, emb):
        return self.clustering_strategy.cluster(emb)
    
    def reduce_emb(self, emb):
        return self.reduction_strategy.reduce_dimensions(emb)

    def process_embedding(
            self, 
            embeddings, section_list, 
            return_mode = ''
        ) -> Optional[str]:
        """
        Оптимизированный пайплайн визуализации
        
        Параметры:
            embeddings, section_list : Входные данные

        Возвращает:
            Обработанные эмбеддинги
        """

        # Обработка данных
        emb_reduce = self.reduction_strategy.reduce_dimensions(embeddings)
        emb_reduce_one_hot = self.add_one_hot(embeddings=emb_reduce, section_list=section_list)
        cluster_reduce_one_hot = self.clustering_strategy.cluster(emb_reduce)

        if return_mode == 'emb_reduce_one_hot':
            result_emb = emb_reduce_one_hot
        elif return_mode == 'emb_reduce':
            result_emb = emb_reduce
        elif return_mode == 'vis_emb':
            result_emb = self.reduction_strategy.reduce_dimensions(emb_reduce_one_hot)
        else:
            result_emb = embeddings
        
        return result_emb, cluster_reduce_one_hot
    
if __name__ == "__main__":
    pass