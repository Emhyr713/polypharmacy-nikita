import re
import json

import sys
sys.path.append("")
from extract_data_from_instructions.extract_data_from_text import ContraindicationExtractor

def extract_pharm_group_and_atc(text: str):
    result = {}

    # Поддержка кириллических и латинских букв в "АТХ"
    atc_header_pattern = r'Код\s+[АA][ТT][ХX]:'

    # Гибкий паттерн для начала фармакодинамического описания
    mech_pattern = (
        r'Механизм\s+действия'
        r'(?:\s+и\s+фармакодинамические\s+(?:свойства|эффекты))?'
        r'|Фармакодинамические\s+эффекты'
    )

    # === 1. Извлекаем фармакотерапевтическую группу ===
    pharm_match = re.search(
        rf'Фармакотерапевтическая\s+группа:\s*(.*?)(?=\s*(?:{atc_header_pattern}|{mech_pattern}|\Z))',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if pharm_match:
        raw = pharm_match.group(1).strip()
        cleaned = re.sub(r'\s+', ' ', raw)
        # Разделяем по ";" и убираем лишние пробелы вокруг каждого элемента
        pharm_parts = [part.strip() for part in cleaned.rstrip('.').split(';')]
        # Удаляем пустые строки, если есть
        pharm_parts = [part for part in pharm_parts if part]
        result['pharm_group'] = pharm_parts  # Теперь это список

    # === 2. Пытаемся найти Код АТХ ===
    atc_full_match = re.search(
        rf'{atc_header_pattern}\s*([A-ZА-Я][0-9][A-ZА-Я0-9]{{2,6}})',
        text,
        re.IGNORECASE
    )
    if atc_full_match:
        result['atc_code'] = atc_full_match.group(1).strip()
        atc_end = atc_full_match.end()
        next_newline = text.find('\n', atc_end)
        content_start = next_newline + 1 if next_newline != -1 else atc_end
    else:
        mech_match = re.search(mech_pattern, text, re.IGNORECASE)
        if mech_match:
            content_start = mech_match.start()
        else:
            content_start = pharm_match.end() if pharm_match else 0

    # === 3. Извлекаем оставшийся текст ===
    remaining = text[content_start:].strip()
    if remaining:
        cleaned_remaining = re.sub(
            rf'^{mech_pattern}\s*',
            '',
            remaining,
            count=1,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()
        result['Фармакодинамические свойства'] = cleaned_remaining or remaining

    return result

def split_text_by_side_effects(text):

    DIR_LIST_CLASS_SIDE_E = "make_side_effect_dataset\\data\\list_class_side_e_edit.txt"

    # Извлечение списка классов побочных эффектов
    with open(DIR_LIST_CLASS_SIDE_E, "r", encoding="utf-8") as file:
        side_e_class_list = [line.strip() for line in file if line.strip()]
    
    # Этап 1: Разделение текста на главы
    pattern = r"^(" + "|".join(re.escape(s) for s in side_e_class_list) + r")\s*"
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
            if part in side_e_class_list:
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
                        re.sub(r'[\s:0-9]+$', '',  # Удаляем числа в конце
                            re.sub(r'^[^a-zA-Zа-яА-ЯёЁ«»]+', '', item)  # Удаляем все не-буквы в начале
                        ).strip()
                        for item in re.split(r'[;,]\s*', sub_part)  # Разделяем по запятой и точке с запятой
                        if item.strip() and not item.strip().isdigit()
                        # and not item.lstrip().startswith('в том числе')
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

if __name__ == "__main__":
    JSON_FILENAME = "OHLP_LV\\data\\OHLP_all.json"
    with open(JSON_FILENAME, 'r', encoding='utf-8') as file:
        drugs = json.load(file)

    drugs_kept = drugs["kept"]
    for drug in drugs_kept:
        # Фармакодинамика
        farm_d_chapter = drug["chapters"]["Фармакодинамические свойства"]
        group_code_text = extract_pharm_group_and_atc(farm_d_chapter)
        drug["chapters"]["Фармакодинамические свойства"] = group_code_text.get("Фармакодинамические свойства", None)
        drug["atc_code"] = group_code_text.get("atc_code", None)
        drug["pharm_group"] = group_code_text.get("pharm_group", None)

        # Побочки
        side_e_chapter = drug["chapters"]["Нежелательные реакции"]
        if side_e_chapter != "":
            side_e_sections = split_text_by_side_effects(side_e_chapter)
            drug["side_e"] = side_e_sections

        # Противопоказания
        contraindications_chapter = drug["chapters"]["Противопоказания"]
        if contraindications_chapter != "":
            extractor = ContraindicationExtractor()
            extracted_contraindication = extractor.extract(contraindications_chapter)
            drug["contraindication"] = extracted_contraindication

    # Сохраняем в JSON
    JSON_FILENAME_RES = "OHLP_LV\\data\\OHLP_all_kept.json"
    with open(JSON_FILENAME_RES, "w", encoding="utf-8") as json_file:
        json.dump(drugs_kept, json_file, ensure_ascii=False, indent=4)