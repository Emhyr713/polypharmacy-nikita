# NormalizeListEdges.py

import re
from pathlib import Path
from typing import Dict, List, Optional, Union



import pandas as pd
import numpy as np
from collections import Counter

import sys
sys.path.append("")
from utils.logger import get_logger
logger = get_logger(__name__)

from create_graph.SemanticEmbeddingProcessor import SemanticEmbeddingProcessor


class NormalizeListEdges:
    """
    Класс для нормализации терминов в графе (source и target) с использованием
    SemanticTermNormalizer на основе SentenceTransformer.
    Удаляет аннотации в скобках и группирует синонимичные термины.
    """

    def __init__(
        self,
        embedding_processor: SemanticEmbeddingProcessor,
        canonical_strategy: str = "most_frequent"
    ):
        self.embedding_processor = embedding_processor
        self.canonical_strategy = canonical_strategy

    @staticmethod
    def remove_parentheses(text: Union[str, float]) -> Union[str, float]:
        """
        Удаляет всё, что в скобках (включая скобки). Обрабатывает NaN.
        Возвращает очищенную строку или np.nan, если вход пустой или не строка.
        """
        if pd.isna(text) or not isinstance(text, str):
            return np.nan
        cleaned = re.sub(r"\([^)]*\)", "", text).strip()
        return cleaned if cleaned else np.nan
    
    def _get_canonical_term(self, cluster, strategy, full_series):
        if strategy == "most_frequent":
            freq = Counter(full_series[full_series.isin(cluster)])
            return freq.most_common(1)[0][0]
        elif strategy == "shortest":
            return min(cluster, key=len)
        else:
            raise ValueError("strategy must be 'shortest' or 'most_frequent'")

    def _aggregate_edges(self, df):
        def mode_or_first(series):
            m = series.mode()
            return m.iloc[0] if not m.empty else series.iloc[0]

        agg_dict = {
            col: mode_or_first if col in ["source_tag", "target_tag"]
            else "first" for col in df.columns if col not in ["source", "target"]
        }
        return df.groupby(["source", "target"], as_index=False).agg(agg_dict)

    def normalize_list_edges(self,
                             df_edges:pd.DataFrame,
                             similarity_threshold: float = 0.95
                             ) -> pd.DataFrame:
        """
        Основной метод: очистка и нормализация source и target
        с сохранением всех остальных колонок.
        Теги (source_tag, target_tag) выбираются по
        наиболее частому значению в группе после нормализации.

        :return: Очищенный и нормализованный DataFrame с агрегированными тегами.
        """
        logger.info("Начало нормализации рёбер с агрегацией тегов по моде")

        mask_list = ["prepare"]

        # Копирование 
        df_work = df_edges.copy()

        # Проверка DataFrame
        required_cols = {"source", "target", "source_tag", "target_tag"}
        if not required_cols.issubset(df_work.columns):
            raise ValueError(f"DataFrame должен содержать колонки: {required_cols}")

        # Очистка скобок
        df_work["source"] = df_work["source"].apply(self.remove_parentheses)
        df_work["target"] = df_work["target"].apply(self.remove_parentheses)

        # Удаление пустых строк
        df_work.dropna(subset=["source", "target"], inplace=True)

        if df_work.empty:
            logger.warning("После очистки данных не осталось.")
            return pd.DataFrame(columns=df_work.columns)

        # Термины, которые НЕ должны нормализоваться (встречаются в строках с тегом 'prepare')
        mask_prepare = df_work["source_tag"].isin(mask_list) | df_work["target_tag"].isin(mask_list)
        terms_to_exclude = pd.concat([
            df_work.loc[mask_prepare, "source"],
            df_work.loc[mask_prepare, "target"]
        ], ignore_index=True)
        terms_to_exclude_set = set(terms_to_exclude)

        # Термины для кластеризации — все, кроме исключённых
        all_terms = pd.concat([df_work["source"], df_work["target"]], ignore_index=True)
        terms_for_clustering = [t for t in all_terms if t not in terms_to_exclude_set]

        normalized_map = {}

        # Заполняем normalized_map для терминов, которые не участвуют в кластеризации
        for term in terms_to_exclude:
            normalized_map[term] = term

        # Если нет терминов для кластеризации, возвращаем DataFrame без изменений (после очистки)
        if not terms_for_clustering:
            result_df = self._aggregate_edges(df_work)
            return self._postprocess_edges(result_df)

        # embeddings = self.embedding_processor.get_embeddings_batch(terms_for_clustering)
        sim_matrix = self.embedding_processor.cosine_similarity_matrix(
            terms_for_clustering, terms_for_clustering, apply_penalty=True
        )

        used_indices = set()
        for i, term in enumerate(terms_for_clustering):
            if i in used_indices or not term.strip():
                continue

            similar_indices = np.where(sim_matrix[i] >= similarity_threshold)[0]
            new_indices = [idx for idx in similar_indices if idx not in used_indices]

            if not new_indices:
                continue

            cluster = [terms_for_clustering[idx] for idx in new_indices]
            canonical = self._get_canonical_term(
                cluster,
                self.canonical_strategy,
                pd.concat([df_work["source"], df_work["target"]], ignore_index=True)
            )

            for t in cluster:
                normalized_map[t] = canonical

            used_indices.update(new_indices)

        # Применяем нормализацию
        df_work["source"] = df_work["source"].map(normalized_map).fillna(df_work["source"])
        df_work["target"] = df_work["target"].map(normalized_map).fillna(df_work["target"])

        # Агрегация
        result_df = self._aggregate_edges(df_work)
        # Постобработка
        postprocess_result_df = self._postprocess_edges(result_df)

        return postprocess_result_df
    
    def _postprocess_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Запуск постобработки: объединение "
                    "симметричных рёбер и удаление петель")

        if df.empty:
            return df
        
        mapping = {}

        # 1. Найти симметричные пары (A -> B и B -> A)
        edges_set = set(map(tuple, df[["source", "target"]].values))
        symmetric_pairs = []
        seen = set()

        for src, tgt in edges_set:
            pair = tuple(sorted([src, tgt]))
            if src != tgt and pair not in seen and (tgt, src) in edges_set:
                symmetric_pairs.append(pair)
                seen.add(pair)

        if symmetric_pairs:
            logger.info(f"Найдено симметричных пар: {len(symmetric_pairs)}")
            # 2. Замена одного узла в паре на каноничный
            full_series = pd.concat([df["source"], df["target"]], ignore_index=True)

            for a, b in symmetric_pairs:
                canonical = self._get_canonical_term([a, b],
                                                     self.canonical_strategy,
                                                     full_series)
                if a != canonical:
                    mapping[a] = canonical
                if b != canonical:
                    mapping[b] = canonical

            # 3. Применить замены
            df["source"] = df["source"].map(mapping).fillna(df["source"])
            df["target"] = df["target"].map(mapping).fillna(df["target"])

        # 4. Удалить петли (source == target), появившиеся после замены
        old_len = len(df)
        df = df[df["source"] != df["target"]].copy()
        if df.empty:
            return pd.DataFrame(columns=df.columns)
        logger.info(f"Удалено петель после замены: {old_len - len(df)}")

        # 5. Агрегировать оставшиеся рёбра
        df = self._aggregate_edges(df)

        logger.info(f"Постобработка завершена. Объединено узлов: {len(set(mapping))}")
        
        return df

    def save_df(self, df: pd.DataFrame, path: Union[str, Path]) -> None:
        """
        Сохраняет DataFrame в CSV.

        :param df: DataFrame для сохранения.
        :param path: Путь к файлу.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, sep=";", index=False, encoding="utf-8")
        logger.info(f"Файл сохранён: {path}")

# main.py
if __name__ == "__main__":
    
    # Пути
    MODEL_PATH = "train_synonim_model/data/synonym-model_1"
    INPUT_PATH = "process_yEd_graph/data/list_edges_verified_folder.csv"
    OUTPUT_PATH = "process_yEd_graph/data/normalized_list_edges.csv"

    # Загрузка
    emb_processor = SemanticEmbeddingProcessor(model_path=MODEL_PATH)
    df_edges = pd.read_csv(INPUT_PATH, sep=";", header=0, dtype=str).fillna("")

    # Нормализация
    normalizer = NormalizeListEdges(emb_processor)

    result_df = normalizer.normalize_list_edges(df_edges=df_edges,
                                                similarity_threshold=0.96)

    # Сохранение
    normalizer.save_df(result_df, OUTPUT_PATH)

    print("✅ Нормализация завершена и сохранена.")