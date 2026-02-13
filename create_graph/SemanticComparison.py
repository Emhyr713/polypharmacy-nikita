import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging

logger = logging.getLogger(__name__)


class SemanticComparison:
    def __init__(
        self,
        model_path: str,
        normalize_threshold: float = 0.95,
        similarity_threshold: float = 0.95,
        use_postprocessing: bool = True,
        number_penalty: float = 0.05
    ):
        """
        :param model_path: Путь к модели SentenceTransformer.
        :param normalize_threshold: Порог косинусного сходства для кластеризации терминов.
        :param similarity_threshold: Минимальное сходство для сопоставления.
        :param use_postprocessing: Включить пост-обработку (например, штраф за числа).
        :param number_penalty: На сколько уменьшаем схожесть при различии в числах (не больше 1.0).
        """
        self.embedding_cache = EmbeddingCache(model_path)
        self.normalize_threshold = normalize_threshold
        self.similarity_threshold = similarity_threshold
        self.use_postprocessing = use_postprocessing
        self.number_penalty = number_penalty

    def build_embeddings(self, terms: List[str]):
        """Теперь просто сохраняет термины и их векторы из кэша."""
        self.embedding_cache.get_embeddings_batch(terms)
        logger.info(f"Corpus установлен: {len(terms)} терминов")

    def _get_canonical_term(
        self,
        cluster: List[str],
        strategy: str,
        full_series: pd.Series
    ) -> str:
        """
        Выбирает канонический термин из кластера.

        :param cluster: Список терминов-синонимов.
        :param strategy: 'shortest' или 'most_frequent'.
        :param full_series: Полный Series для подсчёта частоты.
        :return: Канонический термин.
        """
        if strategy == "most_frequent":
            freq = Counter(full_series[full_series.isin(cluster)])
            return freq.most_common(1)[0][0]
        elif strategy == "shortest":
            return min(cluster, key=len)
        else:
            raise ValueError("strategy должен быть 'shortest' или 'most_frequent'")

    def _normalize_terms(
        self,
        terms: np.ndarray,
        embeddings: np.ndarray,
        strategy: str,
        full_series: pd.Series
    ) -> Dict[str, str]:
        if len(terms) == 0:
            return {}

        sim_matrix = cosine_similarity(embeddings)

        # Применяем штраф: термины сравниваются сами с собой
        adjusted_sim_matrix = self._apply_number_penalty(sim_matrix, list(terms), list(terms))

        normalized_map = {}
        used_indices = set()

        for i, term in enumerate(terms):
            if i in used_indices or not term.strip():
                continue

            similar_indices = np.where(adjusted_sim_matrix[i] >= self.normalize_threshold)[0]
            cluster_terms = [
                terms[idx] for idx in similar_indices
                if idx not in used_indices and terms[idx].strip()
            ]

            if len(cluster_terms) > 0:
                canonical = self._get_canonical_term(cluster_terms, strategy, full_series)
                for t in cluster_terms:
                    normalized_map[t] = canonical
                    used_indices.update(j for j, x in enumerate(terms) if x == t)

        return normalized_map
    
    def normalize(
        self,
        terms: pd.Series,
        strategy: str = "most_frequent"
    ) -> Dict[str, str]:
        """
        Нормализует термины, кластеризуя по семантической близости.

        :param terms: Series с терминами.
        :param strategy: Стратегия выбора канонического термина.
        :return: Словарь {сырой_термин: канонический_термин}
        """
        unique_terms = terms.dropna().astype(str).unique()
        if len(unique_terms) == 0:
            return {}

        logger.info(f"Кодирование {len(unique_terms)} уникальных терминов...")
        embeddings = self.embedding_cache.get_embeddings_batch(unique_terms)

        return self._normalize_terms(unique_terms, embeddings, strategy, terms)

    def find_similar_terms(
        self,
        queries: List[str],
        corpus_terms: List[str],
        top_k: int = 1
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Находит наиболее похожие термины из corpus_terms для каждого запроса.

        :param queries: Список запросов.
        :param corpus_terms: Список терминов, среди которых ищем.
        :param top_k: Количество лучших совпадений.
        :return: Словарь: {запрос: [{"term": ..., "similarity": ...}, ...]}
        """
        if not queries:
            return {}
        if not corpus_terms:
            return {q: [] for q in queries}

        query_embeddings = self.embedding_cache.get_embeddings_batch(queries)
        corpus_embeddings = self.embedding_cache.get_embeddings_batch(corpus_terms)
        sims_matrix = cosine_similarity(query_embeddings, corpus_embeddings)

        sims_matrix = self._apply_number_penalty(sims_matrix, queries, corpus_terms)

        results = {}
        for i, query in enumerate(queries):
            sims = sims_matrix[i]
            top_indices = np.argsort(sims)[::-1][:top_k]
            matches = [
                {"term": corpus_terms[idx], "similarity": float(sims[idx])}
                for idx in top_indices if sims[idx] >= self.similarity_threshold
            ]
            results[query] = matches

        logger.debug(f"Найдено совпадений: {sum(len(v) for v in results.values())}")
        return results

    def _apply_number_penalty(
        self,
        similarities: np.ndarray,
        queries: List[str],
        candidates: List[str]
    ) -> np.ndarray:
        """
        Применяет штраф за различие в числах к матрице схожести.

        :param similarities: Матрица схожести (query x candidate) или вектор.
        :param queries: Список запросов (или терминов).
        :param candidates: Список кандидатов (или терминов).
        :return: Скорректированная матрица/вектор схожести.
        """
        if not self.use_postprocessing or self.number_penalty <= 0:
            return similarities

        adjusted = similarities.copy()
        query_nums_list = [self._extract_numbers(q) for q in queries]

        for i, (query_nums, query) in enumerate(zip(query_nums_list, queries)):
            for j, cand in enumerate(candidates):
                cand_nums = self._extract_numbers(cand)

                # Если есть числа и они различаются — штраф
                if query_nums and cand_nums and set(query_nums) != set(cand_nums):
                    if adjusted.ndim == 1:
                        # Вектор (один запрос)
                        adjusted[j] = max(0.0, adjusted[j] - self.number_penalty)
                    else:
                        # Матрица (много запросов)
                        adjusted[i, j] = max(0.0, adjusted[i, j] - self.number_penalty)

                    logger.debug(f"Штраф за разные числа: '{query}' vs '{cand}' → {adjusted[i,j] if adjusted.ndim > 1 else adjusted[j]:.3f}")

        return adjusted

    @staticmethod
    def _extract_numbers(s: str) -> List[float]:
        """Извлекает все числа из строки (целые и дробные)."""
        return list(map(float, re.findall(r'\d+\.?\d*', s)))
    

# embedding_cache.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List

class EmbeddingCache:
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)
        self.cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, term: str) -> np.ndarray:
        """Возвращает эмбеддинг для термина (из кэша или после кодирования)."""
        term = term.strip()
        if term not in self.cache:
            self.cache[term] = self.model.encode(term, convert_to_tensor=False)
        return self.cache[term]

    def get_embeddings_batch(self, terms: List[str]) -> np.ndarray:
        unique_terms = list(set(term.strip() for term in terms))
        missing = [t for t in unique_terms if t not in self.cache]
        if missing:
            # Пакетное кодирование — быстрее
            embeddings = self.model.encode(missing, convert_to_numpy=True)
            for term, emb in zip(missing, embeddings):
                self.cache[term] = emb
        return np.array([self.cache[t.strip()] for t in terms])

    def preload_terms(self, terms: List[str]):
        """Предварительная загрузка эмбеддингов для списка терминов."""
        for term in set(terms):
            self.get_embedding(term)

    def size(self):
        return len(self.cache)
    

