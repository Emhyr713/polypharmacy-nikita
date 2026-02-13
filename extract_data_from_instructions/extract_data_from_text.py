import json
import sys
sys.path.append("")

from utils.split_sent_razdel import split_format_text
import re

DATA_FILENAME = "extract_data_from_instructions\\data\\extracted_data_all.json"

class PharmacokineticsExtractor:
    def __init__(self):
        # Компилируем регулярные выражения один раз при инициализации
        self.protein_link_pattern = re.compile(
            r'(?=.*\d+\s*%|.*\d+\s*процент)'  # Число + % или "процент"
            r'(?=.*(?:связ|binding))'           # "связ..." или "binding"
            r'(?=.*(?:белк|protein))',          # "белк..." или "protein"
            re.IGNORECASE
        )
        
        self.elimination_pattern = re.compile(
            r'(?=.*\b(?:TCmax|T1/2|Период\s+полувыведения)\b)'
            r'(?=.*\b(?:составляет|час(?:ов|а|)|минут(?:ы|у|)|секунд(?:ы|у|)|сут(?:ок|ки))\b)',
            re.IGNORECASE
        )

    def _split_pharmacokinetics_text(self, text):
        """Общая предобработка текста для обоих методов"""
        index = text.find("Фармакокинетика")
        if index != -1:
            text = text[index:]
        return split_format_text(text, delete_parentheses_flag=False).split('\n')

    def extract_elimination_periods(self, text):
        """
        Извлечение информации о периодах выведения из данных о препаратах
        """
        sentences = self._split_pharmacokinetics_text(text)
        return [s for s in sentences if self.elimination_pattern.search(s)]

    def extract_protein_binding(self, text):
        """
        Извлечение информации о связи с белками
        """
        sentences = self._split_pharmacokinetics_text(text)
        return [s for s in sentences if self.protein_link_pattern.search(s.lower())]

def extract_dosage_form(dosage_form_text):
    """
    Извлечение информации о лекарственной форме из данных о препаратах.
    """
    # dosage_form_set = set()
    # for drug_data in data:
    #     dosage_form_text = drug_data.get('dosage_form', '')
    
    # Преобразование в нижний регистр
    dosage_form_text = dosage_form_text.lower()
    # Удаление числовых значений с единицами измерения
    pattern = r'\d+(?:\.\d+)?(?:\s*(?:мг|мл|г|%|мкг|ед)\/?\s*(?:мг|мл|г|%|мкг|ед)?)'
    dosage_form_text = re.sub(pattern, '', dosage_form_text)
    # Удаление текста в скобках
    dosage_form_text = re.sub(r'\([^)]*\)', '', dosage_form_text)
    # Удаление знаков пунктуации
    dosage_form_text = re.sub(r'[^\w\s]', '', dosage_form_text)
    # Удаление лишних пробелов
    dosage_form_text = ' '.join(dosage_form_text.split())
    # Замена буквы ё на е
    dosage_form_text = dosage_form_text.replace('ё', 'е')

    # Удаление текста "с ароматом"
    if 'с ароматом' in dosage_form_text:
        dosage_form_text = dosage_form_text[:dosage_form_text.find('с ароматом')]

    #     dosage_form_set.add(dosage_form_text)
    # # dosage_form_set = sorted(dosage_form_set)
    # print(f"dosage forms: {dosage_form_set}")

    return dosage_form_text

class ContraindicationExtractor:
    def __init__(self):
        # Паттерн для поиска парных скобок любого типа
        self.bracket_pattern = r'\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}'
        self.limit_len = 60

    def _remove_brackets_content(self, text):
        """Удаляет содержимое парных скобок, пока не останется ни одной пары."""
        prev = None
        while prev != text:
            prev = text
            text = re.sub(self.bracket_pattern, '', text)
        return text

    def _remove_unmatched_brackets_and_text(self, text):
        """Удаляет текст до/после непарных скобок, а также сами непарные скобки."""
        # Удаляем всё после первой непарной открывающей скобки
        text = re.sub(r'\([^)]*$', '', text)
        text = re.sub(r'\[[^\]]*$', '', text)
        text = re.sub(r'\{[^}]*$', '', text)

        # Удаляем всё до последней непарной закрывающей скобки
        text = re.sub(r'^[^(]*\)', '', text)
        text = re.sub(r'^[^[]*\]', '', text)
        text = re.sub(r'^[^{]*\}', '', text)

        # Удаляем оставшиеся одиночные скобки
        text = re.sub(r'[\(\)\[\]\{\}]', '', text)
        return text

    def _normalize_text(self, text):
        # Замена различных типов тире на обычный дефис
        text = text.replace('\u2013', '-')
        text = text.replace('•', '-')
        text = text.replace('\uf02d', '-')
        text = text.replace('\uf0b7', '-')
        text = re.sub(r'[-–−—]+', '-', text)
        # Убираем лишние пробелы вокруг знаков препинания
        # text = re.sub(r'\s*;\s*', ';', text)
        text = re.sub(r'\s*;', ';', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        return text.strip()

    def _split_by_delimiters(self, text):
        parts = re.split(r'\n-|;\s*-|:\s*-|;\n', text)
        fragments = [p.strip() for p in parts if p.strip()]
        return fragments

    def _should_skip_fragment(self, fragment):
        """Проверяет, содержит ли фрагмент стоп-слова."""
        skip_phrases = [
            "гиперчувствительност",
            "чувствительност",
            "применение",
            "одновременное применение"
        ]
        fragment_lower = fragment.lower()
        return any(phrase in fragment_lower for phrase in skip_phrases)

    def extract(self, text):

        # Удаление лишних скобок
        text = self._remove_brackets_content(text)

        # Этап 1: Нормализация исходного текста
        text = self._normalize_text(text)

        # Этап 2: Разбиение по основным разделителям
        fragments = self._split_by_delimiters(text)
        fragments = [re.sub(r'\s+', ' ', f).strip() for f in fragments if f.strip()]

        # 3. Разбиение на предложения
        final_items = []
        for fragment in fragments:
            sentences = split_format_text(fragment,
                                          filter_line_flag=False,
                                          delete_parentheses_flag=False
                                          ).split('\n')
            
            for sentence in sentences:
                if not sentence:
                    continue

                if self._should_skip_fragment(sentence):
                    print(f"ПРОПУСК ПО СТОП-СЛОВУ: {sentence}")
                    continue

                final_items.append(sentence) 
        
        print("---processed fragments:", final_items)

        # 4. Разбиение по вторичным разделителям: ; : . — и запятой, НО не если между цифрами
        final_items_old = final_items  # сохраняем предыдущее состояние
        final_items = []

        for item in final_items_old:
            # Защищаем запятые между цифрами (с возможными пробелами)
            protected = re.sub(r'(\d)\s*,\s*(\d)', r'\1__COMMA__\2', item)
            # Разделяем по ; : . и запятым, которые НЕ являются защищёнными
            parts = re.split(r'[;:.]|,(?!\s*__COMMA__)', protected)
            for part in parts:
                # Восстанавливаем защищённые запятые
                part = part.replace('__COMMA__', ',')
                part = part.strip()
                if part:
                    final_items.append(part)

        # Этап 5: Очистка от скобок и лишних символов
        final_items = [self._remove_brackets_content(item) for item in final_items]
        final_items = [self._remove_unmatched_brackets_and_text(item) for item in final_items]
        final_items = [re.sub(r'^[^\w\s()]+', '', item) for item in final_items]
        final_items = [re.sub(r'[^\w\s()]+$', '', item) for item in final_items]
        final_items = [item.strip() for item in final_items if item.strip()]

        # Уникальность с сохранением порядка
        seen = set()
        unique_items = []
        for item in final_items:
            # Пропускаем, если содержит стоп-слова
            if self._should_skip_fragment(item):
                print(f"ПРОПУСК ПО СТОП-СЛОВУ: {item}")
                continue

            # Пропускаем, если длиннее self.limit_len символов
            if len(item) > self.limit_len:
                print(f"ПРОПУСК ПО ДЛИНЕ > {self.limit_len}: {item}")
                continue

            if item and item not in seen:
                seen.add(item)
                unique_items.append(item)

        return unique_items


if __name__ == "__main__":

    with open(DATA_FILENAME, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # extract_pharmacokinetics(data)

    # extract_dosage_form(data)

    extractor = ContraindicationExtractor()

    for drug_data in data:
        
        if drug_data.get('drug', '') == "Аллопуринол":

            contraindication = drug_data.get("contraindications", "")
            caution = drug_data.get("caution", "")

            print(f"--- Текст {drug_data.get('drug', [])} ---")

            extracted_contraindication = extractor.extract(contraindication)
            extracted_caution = extractor.extract(caution)

            print("--- Contraindications ---")
            print(f"сontraindication_raw: {contraindication}")
            if isinstance(extracted_contraindication, list):
                for item in extracted_contraindication:
                    print("\t", item)
            else:
                print("\t", extracted_contraindication)

            print("--- Cautions ---")
            print(f"caution_raw: {caution}")
            if isinstance(extracted_caution, list):
                for item in extracted_caution:
                    print("\t", item)
            else:
                print("\t", extracted_caution)

            print()