# SemanticEmbeddingProcessor.py

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Optional, Set
import json

class SemanticEmbeddingProcessor:
    def __init__(self, model_path: str, abbr_dataset_path = None):
        self.model = SentenceTransformer(model_path)
        self.cache: Dict[str, np.ndarray] = {}
        self.abbrev_map = {}
        
        # Генератор словаря: для каждого ключа и варианта создаём пару variant → key
        if abbr_dataset_path:
            with open(abbr_dataset_path, 'r', encoding='utf-8') as file:
                abbr_dataset = json.load(file)
            self.abbrev_map = {
                variant.strip(): key.strip()
                for key, variants in abbr_dataset.items()
                for variant in variants  # variants гарантированно список
            }
            print("Загружен датасет аббревиатур")
        else:
            print("Датасет аббревиатур не обнаружен. Замены не будет")

    def get_embedding(self, term: str) -> np.ndarray:
        term = term.strip()
        if term not in self.cache:
            self.cache[term] = self.model.encode(term,
                                                 convert_to_numpy=True,
                                                 show_progress_bar=True)
        return self.cache[term]

    def get_embeddings_batch(self, terms: List[str]) -> np.ndarray:
        unique_terms = list(set(term.strip() for term in terms))
        missing = [t for t in unique_terms if t not in self.cache]
        if missing:
            embeddings = self.model.encode(missing,
                                           convert_to_numpy=True,
                                           show_progress_bar=True)
            for term, emb in zip(missing, embeddings):
                self.cache[term] = emb
        return np.array([self.cache[t.strip()] for t in terms])

    def cosine_similarity_matrix(self,
                                queries: List[str],
                                candidates: List[str],
                                apply_penalty = False,
                                ) -> np.ndarray:
        # Очищаем и нормализуем запросы и кандидаты
        cleaned_queries = [self._clean_term(q) for q in queries]
        cleaned_candidates = [self._clean_term(c) for c in candidates]

        # Получаем эмбеддинги для очищенных строк
        query_embs = self.get_embeddings_batch(cleaned_queries)
        candidate_embs = self.get_embeddings_batch(cleaned_candidates)

        # Считаем косинусное сходство
        sims_matrix = cosine_similarity(query_embs, candidate_embs)

        # Применяем штраф, если нужно
        if apply_penalty:
            sims_matrix = self._apply_number_penalty(
                sims_matrix, cleaned_queries, cleaned_candidates, penalty=0.05
            )

        return sims_matrix
    
    def _clean_term(self, term: str) -> str:
        """
        Удаляет содержимое в скобках, включая скобки, и расшифровывает аббревиатуры.
        """
        term = str(term).lower()
        # Удаляем всё, что в скобках (включая круглые скобки)
        term = re.sub(r'\([^)]*\)', '', term).strip()
        # Раскрываем аббревиатуры, если карта задана
        if self.abbrev_map:
            term = self._expand_abbreviations(term)
        return term
    
    def _expand_abbreviations(self, phrase: str) -> str:
        """
        Заменяет все аббревиатуры, символы или сокращения из self.abbrev_map
        на их полные формы (ключи).
        Учитывает границы слов: заменяет 'β', 'alpha' → 'бета', 'альфа'
        только как отдельные единицы.
        Работает регистронезависимо.

        :param phrase: входная фраза (например, "блокатор β-рецепторов")
        :return: фраза с заменёнными аббревиатурами (например, "блокатор бета-рецепторов")
        """
        if not self.abbrev_map:
            return phrase

        # Сортируем варианты по длине (сначала длинные — чтобы избежать частичных замен)
        sorted_variants = sorted(self.abbrev_map.keys(), key=len, reverse=True)
        result = phrase

        for variant in sorted_variants:
            full_form = self.abbrev_map[variant]
            escaped = re.escape(variant)
            # Проверяем границы слов (чтобы не заменять части слов)
            pattern = rf'(?<![a-zA-Zа-яА-Я]){escaped}(?![a-zA-Zа-яА-Я])'
            result = re.sub(pattern, full_form, result, flags=re.IGNORECASE)

        # if result != phrase:
        #     print(f"Заменена строка: '{phrase}' -> '{result}'")

        return result

    @staticmethod
    def _extract_numbers(s: str) -> List[float]:
        """
        Извлекает все числа (целые и дробные) из строки.
        :param s: входная строка
        :return: список чисел как float
        """
        return [float(match) for match in re.findall(r'\d+\.?\d*', s)]

    @staticmethod
    def _extract_minerals(text: str) -> Set[str]:
        """
        Извлекает минералы из текста с учётом падежей и окончаний.
        Поддерживаемые: калий, магний, натрий, кальций, железо, цинк, медь, селен, марганец, хром.

        Использует регулярные выражения, не требует внешних библиотек.
        """
        # Удаляем содержимое в скобках и приводим к нижнему регистру
        cleaned = re.sub(r'\([^)]*\)', '', text.lower())

        # Паттерны для каждого минерала — с учётом типичных форм (падежи, числа)
        mineral_patterns = {
            'калий':     r'\bкали[еиьяюймо][м]?\b',
            'магний':    r'\bмагни[еиьяюймо][м]?\b',
            'натрий':    r'\bнатри[еиьяюймо][м]?\b',
            'кальций':   r'\bкальц[иеиьяюймо][м]?\b',
            'железо':    r'\bжелез[аоиью][м]?\b',
            'цинк':      r'\bцинк[аеуом]?\b',
            'медь':      r'\b(?:медь|меди|медью)\b',
            'селен':     r'\bселен[аеуом]?\b',
            'марганец':  r'\bмарган[еиьяюйцеом][м]?\b',
            'хром':      r'\bхром[аеуом]?\b'
        }

        found = set()
        for mineral, pattern in mineral_patterns.items():
            if re.search(pattern, cleaned):
                found.add(mineral)
        return found

    def _has_number_mismatch(self, query: str, candidate: str) -> bool:
        """
        Проверяет, есть ли несовпадение чисел (например, дозировок).
        Штрафуется, если оба содержат числа, но они разные.
        """
        nums_q = self._extract_numbers(query)
        nums_c = self._extract_numbers(candidate)
        # Если оба содержат числа, но множества отличаются — несовпадение
        return bool(nums_q and nums_c and set(nums_q) != set(nums_c))

    def _apply_number_penalty(
        self,
        similarities: np.ndarray,
        queries: List[str],
        candidates: List[str],
        penalty: float = 0.05
    ) -> np.ndarray:
        """
        Применяет правила штрафования к матрице сходства:
        - Несовпадение чисел (дозировка)
        - Несовпадение минералов (калий, магний и т.д.)

        Правила легко дополнять внутри функции.

        :param similarities: матрица сходства (n_queries, n_candidates)
        :param queries: список запросов
        :param candidates: список кандидатов
        :param penalty: величина штрафа (по умолчанию 0.05)
        :return: скорректированная матрица сходства
        """
        if penalty <= 0 or similarities.size == 0:
            return similarities

        adjusted = similarities.copy()
        n_queries, n_candidates = adjusted.shape if adjusted.ndim == 2 else (1, len(adjusted))

        # Предварительно очищаем и извлекаем признаки (можно кэшировать при большом объёме)
        query_minerals_list = [self._extract_minerals(q) for q in queries]
        candidate_minerals_list = [self._extract_minerals(c) for c in candidates]

        for i, query in enumerate(queries):
            for j, candidate in enumerate(candidates):
                current_sim = adjusted[i, j] if adjusted.ndim == 2 else adjusted[j]
                apply_penalty_flag = False

                # --- Блок правил ---
                # Правило 1: несовпадение чисел
                if self._has_number_mismatch(query, candidate):
                    # print(f"Найдено несовпадение чисел в паре:'{query}', '{candidate}'")
                    apply_penalty_flag = True

                # Правило 2: несовпадение минералов
                q_min = query_minerals_list[i]
                c_min = candidate_minerals_list[j]
                if q_min and c_min and q_min != c_min:
                    # print(f"Найдено несовпадение элементов в паре:'{query}', '{candidate}'")
                    apply_penalty_flag = True

                # Правило 3: добавление нового правила здесь ↓
                # if self._some_other_rule(query, candidate):
                #     apply_penalty_flag = True

                # --- Конец блока правил ---

                if apply_penalty_flag:
                    current_sim = max(0.0, current_sim - penalty)

                # Сохраняем результат
                if adjusted.ndim == 2:
                    adjusted[i, j] = current_sim
                else:
                    adjusted[j] = current_sim

        return adjusted
    

    def find_similar_terms(
        self,
        queries: List[str],
        corpus_terms: List[str],
        similarity_threshold,
        top_k: int = 1
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Находит наиболее похожие термины из corpus_terms для каждого запроса.

        :param queries: Список запросов.
        :param corpus_terms: Список терминов, среди которых ищем
        :param similarity_threshold: Порог синонимичности
        :param top_k: Количество лучших совпадений.
        :return: Словарь: {запрос: [{"term": ..., "similarity": ...}, ...]}
        """
        if not queries:
            return {}
        if not corpus_terms:
            return {q: [] for q in queries}

        sims_matrix = self.cosine_similarity_matrix(queries, corpus_terms, True)

        results = {}
        for i, query in enumerate(queries):
            sims = sims_matrix[i]
            top_indices = np.argsort(sims)[::-1][:top_k]
            matches = [
                {"term": corpus_terms[idx], "similarity": float(sims[idx])}
                for idx in top_indices if sims[idx] >= similarity_threshold
            ]
            results[query] = matches

        print(f"Найдено совпадений: {sum(len(v) for v in results.values())}")
        return results
    

    def cluster_similar_strings(self,
                                strings_list: list[str],
                                similarity_threshold: float = 0.98
                                ) -> dict[str, list[str]]:
        """
        Кластеризация списка строк по семантической схожести.
        
        :param strings_list: Список строк для кластеризации
        :param similarity_threshold: Порог схожести для объединения в кластер
        :return: Словарь, где ключ - каноническая строка кластера,
                значение - список строк в кластере
        """
        if not strings_list:
            return {}
        
        # Уникализация и очистка списка
        unique_strings = list(dict.fromkeys(strings_list))  # сохраняем порядок первого появления
        unique_strings = [s.strip() for s in unique_strings if s and str(s).strip()]
        
        if not unique_strings:
            return {}
        
        # Получаем матрицу схожести
        sim_matrix = self.cosine_similarity_matrix(
            unique_strings, 
            unique_strings, 
            apply_penalty=True
        )
        
        # Словарь для результатов
        cluster_dict = {}
        used_indices = set()
        
        for i, term in enumerate(unique_strings):
            if i in used_indices:
                continue
            
            # Находим похожие строки
            similar_indices = np.where(sim_matrix[i] >= similarity_threshold)[0]
            new_indices = [idx for idx in similar_indices if idx not in used_indices]
            
            if not new_indices:
                # Если нет похожих строк, создаем кластер из одного элемента
                cluster_dict[term] = [term]
                used_indices.add(i)
                continue
            
            # Формируем кластер
            cluster = [unique_strings[idx] for idx in new_indices]
            
            # Определяем каноническую строку для кластера
            # Используем частоту в исходном списке для выбора наиболее популярной
            # term_counts = {t: strings_list.count(t) for t in cluster}
            # canonical = max(term_counts.items(), key=lambda x: x[1])[0]
            
            # Альтернативный вариант: использовать первую строку в кластере
            canonical = cluster[0]
            
            cluster_dict[canonical] = cluster
            used_indices.update(new_indices)
        
        return cluster_dict

    def size(self) -> int:
        return len(self.cache)
    

if __name__ == "__main__":
    FILENAME_SIDE_E_DICT = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"
    FILENAME_DATASET_ORLOV = "bayes_network\\data\\Orlov.json"
    MODEL_PATH = "train_synonim_model\\data\\synonym-model_4"
    ABBR_MAP_PATH = "create_graph\\data\\abbrev_dict.json"

    import json

    with open(FILENAME_SIDE_E_DICT, "r", encoding="utf-8") as file:
        side_e_list = list(json.load(file)[0])
    # print("side_e_list:", side_e_list)

    with open(FILENAME_DATASET_ORLOV, "r", encoding="utf-8") as file:
        prepare_list = list(json.load(file))
    # print("prepare_list:", prepare_list)

    # print("reference_list:", reference_list)

    prepare_lemm_list = ["изосорбид динитрат", "гепарин натрий"]
    side_e_lemm_list = ["отек квинки", "головной боль", "отек сустав",
                        "аллергический отёк",
                        'гиперчувствительность, включая ангиоотечь',
                        'аллергический отёк и ангионевротический отёк']
    
    other_exapmle = [
        "альфа-адреномиметическое действие", "α-адреномиметическое действие",
        "β-адреномиметическое действие", "α-адреномиметическое действие",
        "бета-адреномиметическое действие", "альфа-адреномиметическое действие",
        "бета-адреномиметическое действие", "β-адреномиметическое действие"
    ]

    queries_list  = prepare_lemm_list + side_e_lemm_list + other_exapmle
    reference_list = side_e_list + prepare_list + other_exapmle
    print()
    semantic_comp = SemanticEmbeddingProcessor(MODEL_PATH,
                                               abbr_dataset_path=ABBR_MAP_PATH)
    semantic_res = semantic_comp.find_similar_terms(queries_list, reference_list,
                                                    0.85, 2)
    print("semantic_res", semantic_res)