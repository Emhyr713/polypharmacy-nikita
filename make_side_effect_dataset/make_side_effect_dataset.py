import copy
import csv
import json
import re
import os
import sys
sys.path.append("")

from utils.fetch_url_page import fetch_page_with_retries
from utils.extract_text_from_pdf import extract_text_from_pdf
from utils.remove_bracket_text import remove_brackets
from utils.translate_by_dictionary import translate

from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer
custom_morph = EnhancedMorphAnalyzer()

# Пути файлов к данным
LINKS_SIDE_EFFECT_FILENAME = "make_side_effect_dataset\\data\\drugs_table4.csv"
INSTRUCTION_DIR_TEMPLATE = "data\\Инструкции_ГРЛС_"
DIR_LIST_CLASS_SIDE_E = "make_side_effect_dataset\\data\\list_class_side_e_edit.txt"
TRANSLATE_DICT = "make_side_effect_dataset\\data\\side_effects_dict_translation.json"
SYNONIM_DICT = ""

# Пути сохранения
SIDE_E_DATASET_FILENAME = 'make_side_effect_dataset\\data\\side_e_dataset.json'
SIDE_E_FREQ_DATASET_FILENAME = 'make_side_effect_dataset\\data\\sef_dataset.json'
SIDE_E_UNIQ_LIST = "make_side_effect_dataset\\data\\sef_uniq_list.txt"

class fetch_pdf_or_rlsnet_side_effects():
    def __init__(self, translate_dict, side_e_class_list, synonim_dict=None):
        self.translate_dict = translate_dict
        self.side_e_class_list = sorted(side_e_class_list, key=len, reverse=True)
        self.synonim_dict = synonim_dict

        self.ND_drug = []
    
    def clear_text(self, text):
        text = text.replace("\r\n", " ")
        text = text.replace("\xa0", " ")
        text = text.replace("ё", "е")
        text = re.sub(r"\(см\.[^)]*\)", "", text)
        text = remove_brackets(text)
        text = text.replace("*", "")
        text = text.replace("§", "")
        text = re.sub(r'\s*/\s*', '/', text)
        text = re.sub(r'\s*:\s*', ':', text)
        text = re.sub(r',\s*$', '', text)
        text = re.sub(r'[^\S\n]+', ' ', text)

        # Удаление строк короче 5 символов (не считая пробелы)
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) >= 5]
        text = '\n'.join(lines)
        return text
    
    def _get_side_effects_rlsnet_text(self, url_part: str) -> dict:
        """Извлекает текст о побочных эффектах с сайта rlsnet.ru."""
        url = "https://www.rlsnet.ru" + url_part
        soup = fetch_page_with_retries(url)

        if not soup:
            return None

        # Удаляем все теги <sup>
        for sup in soup.find_all('sup'):
            sup.decompose()
        
        # Находим секцию с побочными эффектами
        effects_section = soup.find('h2', id=lambda x: x in ['pobocnye-deistviia', 'pobochnie-deistvia'])
        if not effects_section:
            print(f"Не найдена секция с побочными эффектами: {url}")
            return None
        
        for div in effects_section.find_next_siblings('div'):
            # Проверяем условия для style или class
            if ('style' in div.attrs and 'overflow-wrap: break-word' in div['style']) or \
            ('class' in div.attrs and 'text-break' in div['class']):
                target_div = div
                break
        
        if target_div:
            # Извлекаем текст с сохранением переносов строк
            effects_text = target_div.get_text(separator='\n', strip=True)
        else:
            print(f"Не найден div с нужным стилем: {url}")
            return None
        
        effects_text = self.clear_text(effects_text)

        return effects_text
    
    def _get_side_effects_instruction_text(self, extracted_text):
        """
        Извлечение главы с побочными эффектами из текста инструкции.
        Работает только текстовыми pdf.Побочные действия
        """
        # Извлекаем текст после "Побочное действие" и до "Передозировка"
        match_side_e = re.search(
            r'(\nПобочн(?:ое|ые)\s*действ(?:ие|ия)|\nПОБОЧН(?:ОЕ|ЫЕ)\s*ДЕЙСТВ(?:ИЕ|ИЯ))(?:\s*[.:]\s*)?(.*?)(?:\n|\A)(Передозировка|ПЕРЕДОЗИРОВКА)',
            extracted_text,
            re.DOTALL
        )

        if match_side_e:
            text_main = match_side_e.group(2).strip()
            text_main = self.clear_text(text_main)
            return text_main
        else:
            return None
        
    def _extract_text_instruction(self, stage, drug, scan_dict):
        """
        Извлекает текст из PDF или словаря сканов
        """
        text_instruction = None

        if stage in ("1", "2"):
            path = f"{INSTRUCTION_DIR_TEMPLATE}{stage}\\{drug}.pdf"
            if os.path.exists(path):
                text_instruction = extract_text_from_pdf(path)
            else:
                print("Файл не найден:", path)

        elif stage in ("1_scan", "2_scan"):
            if drug in scan_dict:
                text_instruction = scan_dict[drug]
            else:
                print("Не нашли drug в scan_dict")

        return text_instruction

    def split_text_by_side_effects(self, text):
        
        # Этап 1: Разделение текста на главы
        pattern = r"^(" + "|".join(re.escape(s) for s in self.side_e_class_list) + r")\s*"
        parts = re.split(pattern, text, flags=re.MULTILINE)
        
        sections = {}
        current_section = None
        
        # Этап 1: Разделение на главы
        for part in parts:
            if part:
                part = part.replace("\n", " ").strip()  # Убираем символы новой строки из заголовков
                
                if not part:
                    continue
                
                # Пропускаем, если заголовок уже существует
                if part in sections:
                    continue
                
                # Если часть совпадает с заголовком, создаем новую главу
                if part in self.side_e_class_list:
                    current_section = part
                    sections[current_section] = ""
                elif current_section is not None:
                    # Добавляем текст к текущей главе
                    sections[current_section] += (" " if sections[current_section].lower() else "") + part

        # Этап 2: Разделение на подглавы
        sub_sections_list = [
            "частота не известна", "частота неизвестна",
            "очень часто", "очень частые",
            "очень редко", "очень редкие",
            "часто", "частые",
            "редко", "редкие",
            "нечасто", "нечастые",
        ]
        
        sub_pattern = r"(\b" + "|".join(re.escape(s) for s in sub_sections_list) + r"\b)(?=\n|\s*:|-|\*|\d| –| -|\n—|\n-| —|\n:| —| \(| )"
        
        for section, content in sections.items():
            # Разделяем текст на подглавы внутри каждой главы
            sub_parts = re.split(sub_pattern, content.lower(), flags=re.IGNORECASE)
            sub_section = None
            sub_section_content = {}

            for sub_part in sub_parts:
                if sub_part:
                    sub_part = sub_part.strip()

                    if not sub_part:
                        continue

                    # Если подчасть совпадает с подзаголовком, создаем новую подглаву
                    if sub_part.lower() in [s.lower() for s in sub_sections_list]:
                        sub_section = sub_part.lower()
                        sub_section_content[sub_section] = ""

                    # Внутренности подзаголовка
                    elif sub_section is not None:
                        sub_part = sub_part.split(".")[0]  # Разделяем по точке и оставляем первую часть

                        # Удаляем числа и символы (например, ";") в конце строки
                        sub_part = re.sub(r'\s*\d+[\;]?\s*$|[\;\s\)]+$', '', sub_part)

                        sub_part_list = [
                            re.sub(r'\s*\d+$', '',  # Удаляем числа в конце
                                re.sub(r'^[^a-zA-Zа-яА-ЯёЁ«»]+', '', item)  # Удаляем все не-буквы в начале
                            ).strip()
                            for item in re.split(r'[;,]\s*', sub_part)  # Разделяем по запятой и точке с запятой
                            if item.strip() and not item.strip().isdigit() and not item.lstrip().startswith('в том числе')
                        ]

                        sub_section_content[sub_section] = sub_part_list
            
            # Присваиваем подглавы к соответствующей главе
            if sub_section_content:
                sections[section] = sub_section_content
            # Если нет частот
            elif isinstance(sections[section], str):
                sections[section] = sections[section].split(".")[0]
                sections[section] = [
                    re.sub(r'^[^a-zA-Zа-яА-ЯёЁ]+', '', item).strip()
                    for item in re.split(r'[;,]\s*', sections[section])
                    if item.strip()
                ]

        return sections
    
    def get_drugscom_side_e(self, drug_name):

        def extract_frequency_and_effect(text):
            # Регулярное выражение для разделения текста по первой ":"
            pattern = r'^(.*?):\s*(.*)$'
            match = re.match(pattern, text)
            if match:
                frequency = match.group(1).strip()  # Убираем лишние пробелы
                effect = match.group(2).strip()
                return frequency, effect
            else:
                return None

        # Создание ссылки
        prefix = 'https://www.drugs.com/sfx/'
        postfix = '-side-effects.html'
        url = f"{prefix}{drug_name.lower()}{postfix}"       
                
        soup = fetch_page_with_retries(url)

        if not soup:
            return
        
        # Удаляем все теги <sup>
        for sup in soup.find_all('sup'):
            sup.decompose()

        # Найти раздел "For Healthcare Professionals"
        professional_section = soup.find('h2', string='For healthcare professionals')
        if not professional_section:
            print('Раздел "For Healthcare Professionals" не найден.')
            return

        # Инициализировать словарь для хранения побочных эффектов
        side_effects = {}

        # Найти все заголовки h3 и соответствующие списки ul после раздела "For healthcare professionals"
        for element in professional_section.find_all_next(['h2', 'h3', 'ul']):
            # Остановить парсинг при встрече следующего тега h2
            if element.name == 'h2' and element != professional_section:
                break
            # Пропустить элемент
            elif element.name == 'h3':
                current_category = element.get_text(strip=True)
                if current_category == 'General adverse events':
                    continue
                current_category = translate(self.translate_dict, current_category)
                side_effects[current_category] = {}

            elif element.name == 'ul' and current_category:
                for li in element.find_all('li'):
                    full_text = li.get_text(separator=' ', strip=True)
                    full_text = self.clear_text(full_text)
                    full_text = full_text.lower()
                    full_text = extract_frequency_and_effect(full_text)

                    if full_text:
                        frequency, effects = full_text

                        # Перевод частоты
                        frequency = translate(self.translate_dict, frequency)

                        if frequency not in side_effects[current_category]:  
                            side_effects[current_category][frequency] = []

                        effect_list = [translate(self.translate_dict, effect.strip()).lower() for effect in effects.split(",")]

                        side_effects[current_category][frequency].extend(effect_list)

        return side_effects
    
    @staticmethod
    def process_drug_data(dataset):
        """Обрабатывает данные о лекарствах и формирует новую структуру."""
        processed_data = {}

        for drug in dataset:
            if not drug.get("source"):
                continue  # Пропускаем записи без источника

            drug_name_ru = drug["drug_name_ru"].lower()

            # Новая структура данных
            drug_info = {
                "drug_name_en": drug["drug_name_en"],
                "side_e_parts": {},
            }
            side_e_parts = drug.get("side_e_parts")
            if side_e_parts:
                for section, content in side_e_parts.items():
                    if isinstance(content, dict):
                        # Структура: { "часто": [...], "редко": [...] }
                        for freq, effects in content.items():
                            for effect in effects:
                                new_effect = custom_morph.lemmatize_string(effect.lower())
                                drug_info["side_e_parts"][new_effect] = freq
                    elif isinstance(content, list):
                        # Структура: просто список побочек без подзаголовков
                        for effect in content:
                            new_effect = custom_morph.lemmatize_string(effect.lower())
                            drug_info["side_e_parts"][new_effect] = "Частота неизвестна"
                            # print(f"  - {effect}")
                    else:
                        print(f"Неожиданный формат для '{section}': {type(content)}")

            processed_data[drug_name_ru] = drug_info

        return processed_data
    
    @staticmethod
    def get_list_uniq_side_e(json_data):
        uniq_set = set()

        for drug, content in json_data.items():
            for side_e in content['side_e_parts']:
                uniq_set.add(custom_morph.lemmatize_string(side_e.lower()))

        return sorted(uniq_set, key=len, reverse=True)
    
    def convert_side_e(self, side_e_dataset):
        if self.synonim_dict is None:
            print("Словарь не загружен")
            return side_e_dataset
        
        # Глубокая копия всех вложенных структур
        dataset_copy = copy.deepcopy(side_e_dataset)

        convert_dict_map = {word: etalon_word for etalon_word, word in self.synonim_dict.items()}

        for drug in dataset_copy:
            if not drug.get("source"):
                continue

            side_e_parts = drug.get("side_e_parts")
            if side_e_parts:
                for section, content in side_e_parts.items():
                    if isinstance(content, dict):
                        for freq, effects in content.items():
                            new_effects = []
                            for effect in effects:
                                new_effect = custom_morph.lemmatize_string(effect.lower())
                                new_effects.append(convert_dict_map.get(new_effect, new_effect))
                            drug["side_e_parts"][section][freq] = new_effects
                    elif isinstance(content, list):
                        new_effects = []
                        for effect in content:
                            new_effect = custom_morph.lemmatize_string(effect.lower())
                            new_effects.append(convert_dict_map.get(new_effect, new_effect))
                        drug["side_e_parts"][section] = new_effects
                    else:
                        print(f"Неожиданный формат для '{section}': {type(content)}")

        return dataset_copy


if __name__ == "__main__":

    side_e_dataset = []

    # Извлечение списка классов побочных эффектов
    with open(DIR_LIST_CLASS_SIDE_E, "r", encoding="utf-8") as file:
        side_e_class = [line.strip() for line in file if line.strip()]

    # Читаем CSV-файл
    with open(LINKS_SIDE_EFFECT_FILENAME, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile, delimiter=';'))  # Читаем сразу весь файл
        total_rows = len(reader)

    # Извлечение словаря - переводчика
    with open(TRANSLATE_DICT, "r", encoding="utf-8") as file:
        translate_dict = json.load(file)

    # Объединённый словарь инструкций из сканов
    text_instructions_scan = {}
    with open("data\\Инструкции_ГРЛС_1_scan\\Инструкции_ГРЛС_1_scan.json", "r", encoding="utf-8") as file:
        load_data = json.load(file)
        text_instructions_scan.update({item['drug']: item['text'] for item in load_data})
    with open("data\\Инструкции_ГРЛС_2_scan\\Инструкции_ГРЛС_2_scan.json", "r", encoding="utf-8") as file:
        load_data = json.load(file)
        text_instructions_scan.update({item['drug']: item['text'] for item in load_data})

    fetcher = fetch_pdf_or_rlsnet_side_effects(translate_dict, side_e_class)

    for row in reader:
        drug_name_ru = row.get('drug_name_ru', '').strip()
        drug_name_en = row.get('drug_name_en', '').strip()
        stage = row.get('stage', '').strip()

        pdf_filename = row.get('pdf_filename', '').strip()
        drugscom_link = row.get('drugscom_link', '').strip()  
        rlsnet_link = row.get('rlsnet_link', '').strip()
        
        result = {"drug_name_ru": drug_name_ru,
                    "drug_name_en": drug_name_en,
                    "source":None,}
        
        # 1 этап. Извлечение из инструкций
        if pdf_filename and pdf_filename != 'None':
            text_insructions = fetcher._extract_text_instruction(stage, pdf_filename, text_instructions_scan)
            if text_insructions is not None:
                result["source"] = f"{INSTRUCTION_DIR_TEMPLATE}{stage}\\{pdf_filename}.pdf"
                result["text"] = fetcher._get_side_effects_instruction_text(text_insructions)
                if result["text"] is None:
                    print("Не удалось извлечь главу с побочными эффектами:", drug_name_ru, )
                else:
                    result["side_e_parts"] = fetcher.split_text_by_side_effects(result["text"])
            else:
                print("Не удалось извлечь инструкции:", drug_name_ru)

        # 2 этап. Извлечение из rlsnet
        elif rlsnet_link and rlsnet_link != 'None':
            rls_url = f"https://www.rlsnet.ru{rlsnet_link}"
            result["source"] = rls_url
            text = fetcher._get_side_effects_rlsnet_text(rlsnet_link)
            if text:
                result["text"] = fetcher._get_side_effects_rlsnet_text(rlsnet_link)
                result["side_e_parts"] = fetcher.split_text_by_side_effects(result["text"])

        # 3 этап. Извлечение из drugscom
        elif drugscom_link and drugscom_link != 'None':
            drugcom_url = f"https://www.drugs.com/sfx/{drugscom_link}-side-effects.html"      
            result["source"] = drugcom_url
            result["text"] = None
            result["side_e_parts"] = fetcher.get_drugscom_side_e(drugscom_link)

        if not result["source"]:
            fetcher.ND_drug.append(drug_name_ru)
            print("Нет источника побочных эффектов:", drug_name_ru)
        
        side_e_dataset.append(result)

    # Замена побочек на побочку из словаря
    side_e_dataset = fetcher.convert_side_e(side_e_dataset)

    # Конвертация в вид Побочка -- Частота
    sef_dataset = fetcher.process_drug_data(side_e_dataset)

    # Извлечение уникальных побочных эффектов
    sef_uniq_list = fetcher.get_list_uniq_side_e(sef_dataset)

    # Сохранить результаты в JSON-файл
    with open(SIDE_E_DATASET_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(side_e_dataset, f, ensure_ascii=False, indent=4)

    # Сохранить результаты в JSON-файл
    with open(SIDE_E_FREQ_DATASET_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(sef_dataset, f, ensure_ascii=False, indent=4)

    # Сохранить в текстовик
    with open(SIDE_E_UNIQ_LIST, "w", encoding="utf-8") as file:
        file.write("\n".join(sef_uniq_list))