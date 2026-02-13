import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SynonimReplacer:
    """
    Класс для обработки текста: удаление скобок, раскрытие аббревиатур, замена синонимов.
    """

    def __init__(self, model: SentenceTransformer, config_path: str, threshold: float = 0.962):
        """
        :param model: модель SentenceTransformer
        :param config_path: путь к JSON-файлу с 'abbrev' и 'synonyms'
        :param threshold: порог косинусного сходства
        """
        self.model = model
        self.threshold = threshold

        # Загружаем конфиг
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        raw_abbrev = config.get("abbrev", {})
        # Генератор словаря: для каждого ключа и варианта создаём пару variant → key
        self.abbrev_map = {
            variant.strip(): key.strip()
            for key, variants in raw_abbrev.items()
            for variant in variants  # variants гарантированно список
        }

        self.synonym_dict = config.get("synonyms", {})

        # Подготовка синонимов
        self._prepare_synonyms()

    def _prepare_synonyms(self):
        """Предварительная подготовка синонимов и их эмбеддингов."""
        self.all_synonyms = []
        self.synonym_to_key = {}

        for key, synonyms in self.synonym_dict.items():
            for synonym in synonyms:
                self.all_synonyms.append(synonym)
                self.synonym_to_key[synonym] = key

        # Предвычисляем эмбеддинги
        if self.all_synonyms:
            self.synonym_embeddings = self.model.encode(
                self.all_synonyms,
                convert_to_numpy=True
            )
        else:
            self.synonym_embeddings = np.array([]).reshape(0, 768)

    @staticmethod
    def _normalize(text: str) -> str:
        """Нормализация строки: нижний регистр, лишние пробелы."""
        # Удаляем всё между ( и ), включая скобки
        text = re.sub(r'\([^)]*\)', '', text)
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()

    def expand_abbreviations(self, phrase: str) -> str:
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

        for variant in sorted_variants:
            full_form = self.abbrev_map[variant]
            # Экранируем спецсимволы в варианте (например, +, ., *, но не α, β)
            escaped = re.escape(variant)
            # Паттерн: замена, только если вариант стоит как отдельное слово (или с дефисом)
            # (?<!\w) — не после буквы/цифры, (?!\w) — не перед буквой/цифры
            pattern = rf'(?<!\w){escaped}(?!\w)'
            phrase = re.sub(pattern, full_form, phrase, flags=re.IGNORECASE)

        return phrase

    def replace_with_synonyms(self, phrase: str) -> str:
        """
        Ищет в строке подстроки, семантически близкие к синонимам из словаря,
        и заменяет их на соответствующие ключи.

        :param phrase: входная строка (может быть словосочетанием или предложением)
        :return: строка с заменёнными семантическими синонимами
        """
        if not phrase or self.synonym_embeddings.shape[0] == 0:
            return phrase

        words = re.findall(r'\w+', phrase.lower())  # токены без знаков препинания
        if len(words) == 0:
            return phrase

        result = phrase

        # Собираем все возможные непрерывные подстроки (фразы) из слов
        candidates = []
        n = len(words)
        for i in range(n):
            for j in range(i + 1, n + 1):
                phrase_part = ' '.join(words[i:j])
                candidates.append((i, j, phrase_part))

        # Сортируем по длине (сначала длинные — чтобы сначала заменялись составные фразы)
        candidates.sort(key=lambda x: len(x[2]), reverse=True)

        # Храним уже заменённые диапазоны, чтобы не было пересечений
        replaced_spans = []  # (start_word, end_word)

        def overlaps(start, end, spans):
            return any(s < end and start < e for s, e in spans)

        for start_idx, end_idx, candidate_phrase in candidates:
            # Проверяем, не пересекается ли с уже заменённым
            if overlaps(start_idx, end_idx, replaced_spans):
                continue

            # Сравниваем с эмбеддингами синонимов
            input_emb = self.model.encode([candidate_phrase], convert_to_numpy=True).reshape(1, -1)
            similarities = cosine_similarity(input_emb, self.synonym_embeddings).flatten()
            best_idx = np.argmax(similarities)

            if similarities[best_idx] > self.threshold:
                key = self.synonym_to_key[self.synonym_phrases[best_idx]]

                # Находим точное вхождение в оригинальной строке (с сохранением регистра)
                # Используем границы слов
                escaped = re.escape(' '.join(words[start_idx:end_idx]))
                pattern = rf'\b{escaped}\b'
                result = re.sub(pattern, key, result, flags=re.IGNORECASE)

                # Фиксируем, что этот участок уже заменён
                replaced_spans.append((start_idx, end_idx))

        return result
    
if __name__ == "__main__":

    SYNONYM_FILENAME = "data\\dictonary_synonims_simple.json"
    TERM_LIST = "visualization_embedding\\data\\terms_list_verified_folder.json"

    OUTPUT_FILENAME = "train_synonim_model\\data\\find_synonym.csv"

    with open(TERM_LIST, 'r', encoding='utf-8') as file:
        terms_dataset = json.load(file)
    terms = [term[0] for term in terms_dataset]

    # Загрузка модели
    model = SentenceTransformer('train_synonim_model\\data\\synonym-model_1')

    # Создаём обработчик
    processor = SynonimReplacer(model, SYNONYM_FILENAME, threshold=0.95)

    results = []
    for term in terms:
        preprocess_result = processor._normalize(term)
        exp_abbr_result = processor.expand_abbreviations(preprocess_result)
        rpls_result = processor.replace_with_synonyms(exp_abbr_result)

        results.append({
            'raw':term,
            'preprocess_result': preprocess_result,
            'exp_abbr_result': exp_abbr_result,
            'rpls_result': rpls_result
        })

    # Сохранение в CSV
    import csv
    csv_file = 'train_synonim_model\\data\\replace_synonym.csv'
    with open(csv_file, mode='w', encoding='utf-8', newline='',) as file:
        writer = csv.DictWriter(file, fieldnames=['raw',
                                                  'preprocess_result',
                                                  'exp_abbr_result',
                                                  'rpls_result'])
        writer.writeheader()
        writer.writerows(results)

    # test = [
    #     "выведение"
    # ]

    print("Программа успешно завершена")
    