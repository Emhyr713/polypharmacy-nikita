import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import json

class SemanticConnectionChecker:
    def __init__(self, model_path, dataset_path, abbr_dataset = None, sep=';', 
                 normalization_threshold=0.92):
        """
        Инициализация с нормализацией терминов.
        
        Параметры:
            model_path: путь к модели
            dataset_path: путь к CSV (source;target)
            sep: разделитель
            normalization_threshold: порог для объединения терминов (чем выше — строже)
        """
        print("Загрузка модели...")
        self.model = SentenceTransformer(model_path)
        self.normalization_threshold = normalization_threshold
        
        print("Загрузка датасета связей...")
        self.df = pd.read_csv(dataset_path, sep=sep, header=0)

        # Генератор словаря: для каждого ключа и варианта создаём пару variant → key
        if abbr_dataset:
            self.abbrev_map = {
                variant.strip(): key.strip()
                for key, variants in abbr_dataset.items()
                for variant in variants  # variants гарантированно список
            }
            print("Загружен датасет аббревиатур")
        else:
            print("Датасет аббревиатур не обнаружен")
            self.abbrev_map = None
        
        # Очистка от скобок
        self.df['source_clean'] = self.df['source'].apply(self._clean_term)
        self.df['target_clean'] = self.df['target'].apply(self._clean_term)
        
        # Все уникальные очищенные термины
        all_cleaned = pd.concat([self.df['source_clean'], self.df['target_clean']]).dropna().unique()
        self.unique_terms = [t for t in all_cleaned if t.strip()]
        
        print(f"Найдено {len(self.unique_terms)} уникальных терминов. Генерация эмбеддингов...")
        self.term_embeddings = self.model.encode(
            self.unique_terms,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Матрица сходств (N x N)
        print("Вычисление матрицы сходств для нормализации...")
        sim_matrix = cosine_similarity(self.term_embeddings)
        
        # Кластеризация: находим группы похожих терминов
        print("Кластеризация похожих терминов...")
        clusters = self._cluster_similar_terms(sim_matrix)
        
        # Создаём маппинг: старый_термин → канонический_термин
        self.normalization_map = {}
        for cluster in clusters:
            if len(cluster) > 1:
                # Выбираем самый короткий термин как канонический (эвристика)
                # canonical = min(cluster, key=lambda x: len(x))
                canonical = random.choice(cluster)  # случайный выбор из кластера
                for term in cluster:
                    if term != canonical:
                        self.normalization_map[term] = canonical
        
        print(f"Найдено {len(self.normalization_map)} замен для нормализации.")
        
        # Применяем нормализацию к датасету
        self.df['source_normalized'] = self.df['source_clean'].map(
            lambda x: self.normalization_map.get(x, x)
        )
        self.df['target_normalized'] = self.df['target_clean'].map(
            lambda x: self.normalization_map.get(x, x)
        )

        self.embeddings_matrix = np.array(self.term_embeddings)
        
        print("Готово! Проверяльщик инициализирован с нормализованными терминами.")

    def _expand_abbreviations(self, phrase: str) -> str:
        """
        Заменяет все аббревиатуры, символы или сокращения из self.abbrev_map
        на их полные формы (ключи).
        Учитывает границы слов: заменяет 'β', 'alpha' → 'бета', 'альфа'
        только как отдельные единицы.
        Работает с регистронезависимо.

        :param phrase: входная фраза (например, "блокатор β-рецепторов")
        :return: фраза с заменёнными аббревиатурами (например, "блокатор бета-рецепторов")
        """

        # Сортируем варианты по длине (сначала длинные — чтобы избежать частичных замен, например, "al" не заменил "alpha")
        sorted_variants = sorted(self.abbrev_map.keys(), key=len, reverse=True)
        result = phrase

        for variant in sorted_variants:
            full_form = self.abbrev_map[variant]
            escaped = re.escape(variant)
            # Используем negative lookbehind и negative lookahead для проверки границ
            pattern = rf'(?<![a-zA-Zа-яА-Я]){escaped}(?![a-zA-Zа-яА-Я])'
            result = re.sub(pattern, full_form, result, flags=re.IGNORECASE)

        return phrase
    
    def _clean_term(self, term):
        """Удаляет содержимое в скобках, включая скобки, расшифровывает аббревиатуры"""
        term = term.lower()
        term = re.sub(r'\([^)]*\)', '', str(term)).strip()
        if self.abbrev_map:
            term = self._expand_abbreviations(term)
        return term
    
    def _cluster_similar_terms(self, sim_matrix):
        """
        Группирует термины в кластеры по порогу сходства.
        Использует агломеративную логику через граф связей.
        """
        n = len(self.unique_terms)
        visited = np.zeros(n, dtype=bool)
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
            # Новый кластер
            cluster = [self.unique_terms[i]]
            visited[i] = True
            
            # Поиск всех, кто связан с i или с кем-то в кластере
            stack = [i]
            while stack:
                idx = stack.pop()
                for j in range(n):
                    if not visited[j] and sim_matrix[idx, j] >= self.normalization_threshold:
                        cluster.append(self.unique_terms[j])
                        visited[j] = True
                        stack.append(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    

    
    def check_pair(self, term1, term2, similarity_threshold=0.92):
        """
        Проверяет связь между двумя терминами через нормализованный датасет.
        """

        cleaned_term1 = self._clean_term(term1)
        cleaned_term2 = self._clean_term(term2)
        
        # Нормализуем входные термины (если есть в маппинге)
        norm_term1 = self.normalization_map.get(cleaned_term1, cleaned_term1)
        norm_term2 = self.normalization_map.get(cleaned_term2, cleaned_term2)
        
        # Получаем эмбеддинги
        input_embs = self.model.encode([cleaned_term1, cleaned_term2], convert_to_numpy=True)
        emb1 = input_embs[0].reshape(1, -1)
        emb2 = input_embs[1].reshape(1, -1)
        
        # Поиск ближайших в уникальных терминах
        sims_to_1 = cosine_similarity(emb1, self.term_embeddings)[0]
        sims_to_2 = cosine_similarity(emb2, self.term_embeddings)[0]
        
        similar_to_1 = [(self.unique_terms[i], sims_to_1[i]) 
                       for i in range(len(sims_to_1)) if sims_to_1[i] >= similarity_threshold]
        similar_to_2 = [(self.unique_terms[i], sims_to_2[i]) 
                       for i in range(len(sims_to_2)) if sims_to_2[i] >= similarity_threshold]
        
        set1 = {self.normalization_map.get(t, t) for t, s in similar_to_1}
        set2 = {self.normalization_map.get(t, t) for t, s in similar_to_2}
        
        # Проверяем связь в нормализованном датасете
        for _, row in self.df.iterrows():
            src_norm = row['source_normalized']
            tgt_norm = row['target_normalized']
            if src_norm in set1 and tgt_norm in set2:
                sim1 = max([s for t, s in similar_to_1 
                           if self.normalization_map.get(t, t) == src_norm], default=0)
                sim2 = max([s for t, s in similar_to_2 
                           if self.normalization_map.get(t, t) == tgt_norm], default=0)
                return {
                    'has_connection': True,
                    'similar_source': src_norm,
                    'similar_target': tgt_norm,
                    'similarity_term1': sim1,
                    'similarity_term2': sim2,
                    'original_mapping_source': [k for k, v in self.normalization_map.items() if v == src_norm],
                    'original_mapping_target': [k for k, v in self.normalization_map.items() if v == tgt_norm],
                    'input_term1': term1,
                    'input_term2': term2
                }
            if src_norm in set2 and tgt_norm in set1:
                sim2 = max([s for t, s in similar_to_1 
                           if self.normalization_map.get(t, t) == tgt_norm], default=0)
                sim1 = max([s for t, s in similar_to_2 
                           if self.normalization_map.get(t, t) == src_norm], default=0)
                return {
                    'has_connection': True,
                    'similar_source': tgt_norm,
                    'similar_target': src_norm,
                    'similarity_term1': sim1,
                    'similarity_term2': sim2,
                    'original_mapping_source': [k for k, v in self.normalization_map.items() if v == tgt_norm],
                    'original_mapping_target': [k for k, v in self.normalization_map.items() if v == src_norm],
                    'input_term1': term1,
                    'input_term2': term2
                }
        
        # 🔴 Если связь НЕ найдена — всё равно возвращаем синонимы
        return {
            'has_connection': False,
            'input_term1': term1,
            'input_term2': term2,
            'normalized_inputs': (norm_term1, norm_term2),
            'synonyms_term1': [t for t, s in similar_to_1],
            'similarity_scores_term1': [(t, s) for t, s in similar_to_1],
            'synonyms_term2': [t for t, s in similar_to_2],
            'similarity_scores_term2': [(t, s) for t, s in similar_to_2],
            'threshold': similarity_threshold
        }
    

if __name__ == "__main__":

    SYNONYM_FILENAME = "data\\dictonary_synonims_simple.json"
    with open(SYNONYM_FILENAME, 'r', encoding='utf-8') as file:
        abbr_dataset = json.load(file)['abbrev']

    # Инициализация (один раз)
    checker = SemanticConnectionChecker(
        model_path='train_synonim_model\\data\\synonym-model_1',
        dataset_path='process_yEd_graph\\data\\list_edges_verified_folder.csv',
        abbr_dataset=abbr_dataset,
        sep=';',
        normalization_threshold=0.965
    )

    print("\n" + "🔍 Система проверки семантических связей запущена")
    print("Введите 'exit' в любом поле, чтобы выйти\n")

    while True:
        print("-" * 60)
        term_1 = input("Первый термин: ").strip()
        if term_1.lower() == 'exit':
            break
        term_2 = input("Второй термин: ").strip()
        if term_2.lower() == 'exit':
            break

        # === Шаг 1: Показываем, как термины были обработаны и нормализованы ===
        cleaned1 = checker._clean_term(term_1)
        cleaned2 = checker._clean_term(term_2)

        norm1 = checker.normalization_map.get(cleaned1, "не заменён")
        norm2 = checker.normalization_map.get(cleaned2, "не заменён")

        print("\n📝 Обработка ввода:")
        print(f"  '{term_1}' → очищено: '{cleaned1}'")
        if norm1 != "не заменён":
            print(f"\t\t\t→ нормализовано: '{norm1}' (синоним)")
        else:
            print(f"\t\t\t→ без нормализации")

        print(f"  '{term_2}' → очищено: '{cleaned2}'")
        if norm2 != "не заменён":
            print(f"\t\t\t→ нормализовано: '{norm2}' (синоним)")
        else:
            print(f"\t\t\t→ без нормализации")

        # === Шаг 2: Поиск ближайших терминов из датасета (для контекста) ===
        emb1 = checker.model.encode([cleaned1], convert_to_numpy=True).reshape(1, -1)
        emb2 = checker.model.encode([cleaned2], convert_to_numpy=True).reshape(1, -1)

        sims1 = cosine_similarity(emb1, checker.embeddings_matrix)[0]
        sims2 = cosine_similarity(emb2, checker.embeddings_matrix)[0]

        top5_1 = sorted(zip(checker.unique_terms, sims1), key=lambda x: x[1], reverse=True)[:5]
        top5_2 = sorted(zip(checker.unique_terms, sims2), key=lambda x: x[1], reverse=True)[:5]

        print("\n🔍 Ближайшие термины в датасете:")
        print("  Похожие на первый:")
        for t, s in top5_1:
            mark = " ← нормализован" if checker.normalization_map.get(t) == norm1 or t == norm1 else ""
            print(f"\t• {t} (схожесть: {s:.3f}){mark}")

        print("\tПохожие на второй:")
        for t, s in top5_2:
            mark = " ← нормализован" if checker.normalization_map.get(t) == norm2 or t == norm2 else ""
            print(f"\t• {t} (схожесть: {s:.3f}){mark}")

        # === Шаг 3: Проверка связи ===
        result = checker.check_pair(
            term_1,
            term_2,
            similarity_threshold=0.962
        )

        print("\n" + ("✅ СЕМАНТИЧЕСКАЯ СВЯЗЬ НАЙДЕНА!" if result['has_connection'] else "❌ Связь не найдена"))
        print("— " * 30)

        if result['has_connection']:
            print(f"🔗 Найдена связь через нормализованные термины:")
            print(f"   {result['similar_source']}  ───→  {result['similar_target']}")

            print(f"\n📊 Степень схожести:")
            print(f"\t'{term_1}' → '{result['similar_source']}': {result['similarity_term1']:.3f}")
            print(f"\t'{term_2}' → '{result['similar_target']}': {result['similarity_term2']:.3f}")

            if result['original_mapping_source']:
                print(f"\n🔄 Синонимы источника: {', '.join(result['original_mapping_source'])}")
            if result['original_mapping_target']:
                print(f"🔄 Синонимы цели: {', '.join(result['original_mapping_target'])}")

            # print(f"\n📄 Оригинальная связь в датасете:")
            # print(f"   {result['original_source']} → {result['original_target']}")
        else:
            print(f"ℹ️  Не удалось найти прямую связь между:")
            print(f"\t'{result['input_term1']}' и '{result['input_term2']}'.")

            print(f"\n🔍 Найденные синонимы (порог: {result['threshold']:.3f}):")
            if result['synonyms_term1']:
                print(f"  Похожие на '{term_1}':")
                for t, s in result['similarity_scores_term1']:
                    mark = " ← нормализован" if t == result['normalized_inputs'][0] else ""
                    print(f"\t• {t} (схожесть: {s:.3f}){mark}")
            else:
                print(f"  🚫 Нет терминов в датасете, похожих на '{term_1}'")

            if result['synonyms_term2']:
                print(f"  Похожие на '{term_2}':")
                for t, s in result['similarity_scores_term2']:
                    mark = " ← нормализован" if t == result['normalized_inputs'][1] else ""
                    print(f"\t• {t} (схожесть: {s:.3f}){mark}")
            else:
                print(f"  🚫 Нет терминов в датасете, похожих на '{term_2}'")

            print(f"\n💡 Попробуйте использовать один из этих синонимов или проверьте, есть ли смысловая цепочка.")

        print("\n")