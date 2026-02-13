# KnowledgeGraphExplorer.py

from typing import List, Set, Dict, Optional
from collections import deque
import pandas as pd
import numpy as np
import networkx as nx
import uuid



# Настройка логирования
import sys
sys.path.append("")
from utils.logger import get_logger
logger = get_logger(__name__)

from create_graph.SemanticEmbeddingProcessor import SemanticEmbeddingProcessor


class KnowledgeGraphExplorer:
    """
    Класс для построения графа знаний с уникальными ID (UUID), семантического
    сопоставления по именам (с штрафом за числа) и рекурсивного обхода потомков.
    """

    def __init__(
        self,
        embedding_processor: SemanticEmbeddingProcessor
    ):
        self.embedding_processor = embedding_processor
        self.graph = None
        self.node_id_to_name = {}
        self.node_name_to_ids = {}
        self.corpus_terms = []

    def build_graph(self, df) -> 'KnowledgeGraphExplorer':
        """Строит направленный граф, объединяя одинаковые узлы (по имени + тег)."""
        logger.info("Построение направленного графа с унификацией узлов по (name, tag)...")
        
        if(not isinstance(df, pd.DataFrame)
           or 'source' not in df.columns
           or 'target' not in df.columns):
            raise ValueError("DataFrame должен содержать колонки 'source' и 'target'.")
        
        self.graph = nx.DiGraph()
        self.node_id_to_name.clear()
        self.node_name_to_ids.clear()

        # Кэш: ключ (name, tag) → UUID
        node_key_to_id = {}

        for _, row in df.iterrows():
            source = str(row['source']).strip()
            source_tag = str(row['source_tag']).strip()
            target = str(row['target']).strip()
            target_tag = str(row['target_tag']).strip()

            # Ключ для узла: (имя, тег)
            source_key = (source, source_tag)
            target_key = (target, target_tag)

            # Получаем или создаём UUID по ключу
            if source_key not in node_key_to_id:
                node_key_to_id[source_key] = str(uuid.uuid4())
            if target_key not in node_key_to_id:
                node_key_to_id[target_key] = str(uuid.uuid4())

            source_id = node_key_to_id[source_key]
            target_id = node_key_to_id[target_key]

            # Добавляем узлы
            if source_id not in self.graph:
                self.graph.add_node(source_id, label=f"{source}({source_tag})")
            if target_id not in self.graph:
                self.graph.add_node(target_id, label=f"{target}({target_tag})")

            # Добавляем ребро
            self.graph.add_edge(source_id, target_id)

            # Сохраняем соответствие: UUID → имя (без тега)
            self.node_id_to_name[source_id] = source
            self.node_id_to_name[target_id] = target
            self.node_name_to_ids.setdefault(source, []).append(source_id)
            self.node_name_to_ids.setdefault(target, []).append(target_id)

        logger.info(f"Граф построен: "
                    f"{len(self.graph.nodes)} узлов, "
                    f"{len(self.node_name_to_ids)} уникальных имён, "
                    f"{len(self.graph.edges)} рёбер.")

        # Передаём все уникальные имена в normalizer для получения эмбеддингов
        self.corpus_terms = list(self.node_name_to_ids.keys())
        return self

    def _find_starting_node_ids(self, seed_words: List[str],
                                similarity_threshold: float
                                ) -> Set[str]:
        """
        Находит UUID узлов, чьи имена семантически близки к seed_words.
        :param seed_words: Список начальных слов.
        :param similarity_threshold: Порог схожести.
        :return: Множество UUID подходящих узлов.
        """
        if not self.corpus_terms:
            logger.warning("Corpus пуст — нечего сопоставлять.")
            return set()

        sims_matrix = self.embedding_processor.cosine_similarity_matrix(
            seed_words, self.corpus_terms, apply_penalty=True
        )
        matched_ids = set()

        for i, word in enumerate(seed_words):
            sims = sims_matrix[i]

            # Защита от пустого вектора
            if len(sims) == 0:
                logger.warning(f"❌ '{word}' не сопоставлен: "
                               f"пустой результат схожести")
                continue

            # Найти все термины выше порога
            matches = np.where(sims >= similarity_threshold)[0]

            if len(matches) > 0:
                for idx in matches:
                    term = self.corpus_terms[idx]
                    node_ids = self.node_name_to_ids.get(term, [])
                    matched_ids.update(node_ids)
                    logger.info(f"✅ '{word}' → '{term}' "
                                f"(схожесть: {sims[idx]:.3f}), "
                                f"найдено {len(node_ids)} узлов")
            else:
                max_sim = sims.max()
                logger.warning(f"❌ '{word}' не сопоставлен "
                               f"(макс. схожесть: {max_sim:.3f} < "
                               f"{similarity_threshold})")

        return matched_ids

    def _get_all_descendants(self, start_node_ids: Set[str]) -> Set[str]:
        """
        Находит всех потомков (достижимые узлы по UUID) от стартовых узлов.

        :param start_node_ids: Множество UUID стартовых узлов.
        :return: Множество UUID всех достижимых узлов.
        """
        if not start_node_ids:
            return set()

        visited = set(start_node_ids)
        queue = deque(start_node_ids)

        logger.info("Запуск BFS для поиска всех потомков (по UUID, защита от циклов)...")

        while queue:
            node_id = queue.popleft()
            for successor_id in self.graph.successors(node_id):
                if successor_id not in visited:
                    visited.add(successor_id)
                    queue.append(successor_id)

        logger.info(f"Найдено {len(visited)} достижимых узлов (включая стартовые).")
        return visited

    def explore_from_seeds(self,
                           seed_words: List[str],
                           similarity_threshold: float = 0.9
                           ) -> nx.DiGraph:
        """
        Основной метод: находит узлы, близкие к seed_words, и возвращает подграф
        со всеми их потомками.

        :param seed_words: Список начальных слов/фраз.
        :return: Подграф (с UUID в качестве узлов).
        """
        if self.graph is None:
            raise RuntimeError("Граф не построен. Вызовите build_graph().")

        # Шаг 1: Найти стартовые узлы (по UUID)
        start_node_ids = self._find_starting_node_ids(
            seed_words, similarity_threshold
        )
        if not start_node_ids:
            logger.warning("Не найдено ни одного стартового узла. "
                           "Возвращаем пустой граф.")
            return nx.DiGraph()

        # Шаг 2: Найти всех потомков
        descendant_ids = self._get_all_descendants(start_node_ids)
        full_ids = descendant_ids  # можно включить стартовые, они уже в visited

        # Шаг 3: Построить подграф
        subgraph = self.graph.subgraph(full_ids).copy()
        logger.info(f"Финальный подграф: "
                    f"{len(subgraph.nodes)} узлов, "
                    f"{len(subgraph.edges)} рёбер.")
        return subgraph

    def get_nodes(self, graph: nx.Graph) -> List[Dict]:
        """Возвращает узлы с атрибутами для визуализации."""
        return [
            {
                "id": node_id,
                "label": data.get("label", node_id),
            }
            for node_id, data in graph.nodes(data=True)
        ]

    def get_edges(self, graph: nx.Graph) -> List[Dict]:
        """Возвращает рёбра."""
        return [
            {
                "source": u,
                "target": v,
                "relation_type": d.get("relation_type", "related")
            }
            for u, v, d in graph.edges(data=True)
        ]

    def save_graph(self, graph: nx.Graph, path: str) -> None:
        """Сохраняет граф в GraphML."""
        nx.write_graphml(graph, path)
        logger.info(f"Граф сохранён: {path}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Пути
    MODEL_PATH = "train_synonim_model/data/synonym-model_1"
    INPUT_PATH = "process_yEd_graph/data/list_edges_verified_folder.csv"

    df_edges = pd.read_csv(INPUT_PATH, sep=";", header=0, dtype=str).fillna("")

    # === 2. Загрузка нормализатора ===
    normalizer = SemanticEmbeddingProcessor(model_path=MODEL_PATH)

    # === 3. Создание и использование explorer'а ===
    explorer = KnowledgeGraphExplorer(embedding_processor=normalizer)
    explorer.build_graph(df_edges)

    # === 4. Стартовые слова ===
    seed_words = [
        "ингибитор ангетензинопревращать",
    ]

    # === 5. Рекурсивный обход всех потомков ===
    subgraph = explorer.explore_from_seeds(seed_words, 0.9)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
    nx.draw(
        subgraph, pos,
        with_labels=True,
        node_size=800,
        node_color="skyblue",
        font_size=10,
        font_family="DejaVu Sans",
        edge_color="gray",
        arrows=True,
        arrowsize=15,
        alpha=0.9
    )
    plt.title("Граф знаний", fontsize=16, fontfamily="DejaVu Sans")
    plt.tight_layout()
    plt.show()
