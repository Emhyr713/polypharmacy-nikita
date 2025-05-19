from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from abc import ABC, abstractmethod
from typing import List

class VisualizationStrategy(ABC):
    @abstractmethod
    def visualize(self, reduced_embeddings: np.ndarray, labels: List[str], classes: List[str], title: str):
        pass

class MatplotlibVisualization(VisualizationStrategy):
    def visualize(self, reduced_embeddings: np.ndarray, labels: List[str], classes: List[str], title: str):
        plt.figure(figsize=(12, 10))
        
        # Группировка по классам
        unique_classes = list(set(classes))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        
        for cls, color in zip(unique_classes, colors):
            mask = np.array(classes) == cls
            plt.scatter(reduced_embeddings[mask, 0], 
                        reduced_embeddings[mask, 1], 
                        color=color, 
                        label=cls,
                        alpha=0.7)
        
        # Добавление подписей для некоторых точек
        for i, name in enumerate(labels):
            if i % 10 == 0:  # Подписываем каждую 10-ю точку
                plt.annotate(name, 
                            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                            fontsize=8)
        
        plt.title(title)
        plt.xlabel("Компонента 1")
        plt.ylabel("Компонента 2")
        plt.legend()
        plt.grid(True)
        plt.show()

class PlotlyVisualization(VisualizationStrategy):
    def visualize(self, reduced_embeddings: np.ndarray, labels: List[str], classes: List[str], title: str):
        fig = px.scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            color=classes,
            hover_name=labels,
            title=title,
            labels={'color': 'Class'},
            width=1800,
            height=1000
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(legend_title_text='Classes')
        fig.show()
        return fig
    
    def save_visualize(self,
                        reduction_method,
                        clustering_method,
                        clusters,
                        fig,
                        base_dir = "visualization_embedding\\data\\save_pic"):
        n_clusters = len(np.unique(clusters))
        filename = f"{reduction_method}_{clustering_method}_clusters{n_clusters}.html"
        fig.write_html(f"{base_dir}\\{filename}")  # Полностью интерактивный