import re
import json
import os
import sys
sys.path.append("")

from utils.extract_text_from_pdf import extract_text_from_pdf
from extract_data_from_instructions.extract_data_from_text import (extract_dosage_form,
                                                            ContraindicationExtractor,
                                                            PharmacokineticsExtractor)

DIR_LIST_PDF = [
    "data\\Инструкции_ГРЛС_1",
    "data\\Инструкции_ГРЛС_2"
]

# JSON_LIST_SCANS = [
#     # "data\\Инструкции_ГРЛС_1_scan\\Инструкции_ГРЛС_1_scan.json",
#     # "data\\Инструкции_ГРЛС_2_scan\\Инструкции_ГРЛС_2_scan.json",
# ]

DIR_SAVE = "extract_data_from_instructions\\data"

def fix_terms(text):
    """
    Исправляет написание терминов,
    чтобы избежать путаницы (например, Тmax -> Tmax, Сmax -> Cmax).
    """
    if not text:
        return text

    # Словарь замен: кириллические буквы на латинские в известных терминах
    replacements = {
        'Тmax': 'Tmax',
        'Тmin': 'Tmin',
        'Сmax': 'Cmax',
        'Сmin': 'Cmin',
        'Сss': 'Css',
        'Т1/2': 'T1/2',
        'Т½': 'T1/2',
        'Стах': 'Cmax',
        'Cмах': 'Cmax',
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    # Универсальные замены для смешанных вариантов написания
    text = re.sub(r'[ТT][ ]?max', 'Tmax', text, flags=re.IGNORECASE)
    text = re.sub(r'[СC][ ]?max', 'Cmax', text, flags=re.IGNORECASE)
    text = re.sub(r'[СC][ ]?min', 'Cmin', text, flags=re.IGNORECASE)
    text = re.sub(r'[ТT][ ]?min', 'Tmin', text, flags=re.IGNORECASE)
    text = re.sub(r'[ТT][ ]?(1/2|½)', 'T1/2', text, flags=re.IGNORECASE)

    # lines = text.split("\n")

    # Фильтрация строк: оставляем только те, которые НЕ соответствуют шаблону
    # Удалить "С. 14"
    filtered_text = re.sub(r"С(тр?|\.)\s*\d+(\s*из\s*\d+)?", "",
                           text, flags=re.IGNORECASE)
    # Стр.
    # filtered_text = "\n".join([s for s in lines
    #                             if not re.fullmatch(r"С(тр?|\.)\s*\d+\s*из\s*\d+", s)
    #                             ])
    
    # 
    filtered_text = re.sub(
        r'\(см\.\s*(?:под)?раздел[а-я]*\s*[^)]*\)', '', 
        filtered_text, flags=re.IGNORECASE | re.UNICODE
    )
    filtered_text = re.sub(
        r'см\.(?:\s+также)?(?:\s*(?:под)?раздел[а-я]*)?\s*«[^»]{0,80}»', '',
        filtered_text, flags=re.IGNORECASE | re.UNICODE
    )
    filtered_text = re.sub(
        r'\(см\.\s[^)]{0,78}\)', '',
        filtered_text, flags=re.UNICODE
    )

    return filtered_text

def extract_sections(name, text):
    """
    Извлечение разделов из текста инструкции.
    """

    def extract_clean_section(text, pattern, delete_empty=True):
        """
        Удаление лишних пробелов и переносов строк из извлеченного текста.
        """
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(2).strip()
            if delete_empty:
                extracted = re.sub(r'\s+', ' ', extracted)  # Удалить лишние пробелы
                
            return extracted
        return None

    sections = {"drug": name}

    patterns = [
        (
            "group",
            (
                r'(?:\n|^)(Фармакотерапевтическая\s*группа|ФАРМАКОТЕРАПЕВТИЧЕСКАЯ\s*ГРУППА)'
                r'(?:\s*[.:]\s*)?(.*?)'
                r'(?=\s*[‚\-–—]*\s*[КK]\S*?\s+[AА][ТT][ХX])'
            ),
            True
        ),
        (
            "text",
            (
                r'(?:\n|^)(Фармакологические\s*свойства|Фармакологическое\s*действие|ФАРМАКОЛОГИЧЕСКИЕ\s*СВОЙСТВА)'
                r'(?:\s*[.:]\s*)?(.*?)'
                r'(?:\n|\A)'
                r'(Показания\s*к\s*применению|Показания\s*к\s*медицинскому\s*применению|ПОКАЗАНИЯ\s*К\s*ПРИМЕНЕНИЮ)'
            ),
            True
        ),
        (
            "text_side_e",
            (
                r'(?:\n|^)(Побочн(?:ое|ые)\s*действ(?:ие|ия)|ПОБОЧН(?:ОЕ|ЫЕ)\s*ДЕЙСТВ(?:ИЕ|ИЯ))'
                r'(?:\s*[.:]\s*)?(.*?)'
                r'(?:\n|\A)'
                r'(Передозировка|ПЕРЕДОЗИРОВКА)'
            ),
            False
        ),
        (
            "dosage_form",
            (
                r'(?:\n|^)(Лекарственная\s*форма|ЛЕКАРСТВЕННАЯ\s*ФОРМА)'
                r'(?:\s*[.:]\s*)?(.*?)'
                r'(?:\n|\A)'
                r'(Состав|СОСТАВ|Описание|ОПИСАНИЕ)'
            ),
            True
        ),
        (
            "form_release",
            (
                r'(?:\n|^)(Форма\s*выпуска|ФОРМА\s*ВЫПУСКА)'
                r'(?:\s*[.:]\s*)?(.*?)'
                r'(?:\n|\A)'
                r'(Условия\s*хранения|УСЛОВИЯ\s*ХРАНЕНИЯ|Хранение|ХРАНЕНИЕ)'
            ),
            True
        ),
        (
            "contraindications",
            (
                r'(?:\n|^)(Противопоказания|ПРОТИВОПОКАЗАНИЯ)'
                r'(?:\s*[.:]\s*)?(.*?)'
                r'(?:\n|\A)'
                r'(?=\s*С\s*осторожностью|С\s*ОСТОРОЖНОСТЬЮ|'
                r'Применение\s*при\s*беременности\s*и\s*в\s*период\s*грудного\s*вскармливания|'
                r'Способ\s*применения\s*и\s*дозы)'
            ),
            False
        ),
        (
            "caution",
            (
                r'(?:\n|^)(С\s*осторожностью|С\s*ОСТОРОЖНОСТЬЮ)'
                r'(?:\s*[.:]\s*)?(.*?)'
                r'(?:\n|\A)'
                r'(?=\s*Применение\s*при\s*беременности\s*и\s*в\s*период\s*грудного\s*вскармливания|'
                r'ПРИМЕНЕНИЕ\s*ПРИ\s*БЕРЕМЕННОСТИ\s*И\s*В\s*ПЕРИОД\s*ГРУДНОГО\s*ВСКАРМЛИВАНИЯ|'
                r'Способ\s*применения\s*и\s*дозы|Передозировка|ПЕРЕДОЗИРОВКА)'
            ),
            False
        ),
    ]

    for key, pattern, delete_empty in patterns:
        content = extract_clean_section(text,
                                        pattern,
                                        delete_empty=delete_empty)
        if content:
            sections[key] = content

    return sections

def extract_text_pdf(dir_list = DIR_LIST_PDF):
    """
    Извлечение текста из PDF-документов.
    """

    drug_jsones = []
    
    for directory in list(dir_list):

        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                name = os.path.splitext(filename)[0]
                name = name.replace("_OCR", "")
                pdf_path = os.path.join(directory, filename)
                
                extracted_text = extract_text_from_pdf(pdf_path)
                fix_extracted_text = fix_terms(extracted_text)
                drug_sections = extract_sections(name, fix_extracted_text)

                if drug_sections:
                    drug_jsones.append(drug_sections)

    return drug_jsones

# def extract_text_scans(json_list = JSON_LIST_SCANS):
#     """
#     Извлечение текста из сканов.
#     """
#     drug_jsones = []
#     for json_file in list(json_list):
#         with open(json_file, 'r', encoding='utf-8') as file:
#             data = json.load(file)
#         for instructions in data:
#             drug_sections = extract_sections(instructions["drug"],
#                                              instructions["text"])
#             if drug_sections:
#                 drug_jsones.append(drug_sections)
#     return drug_jsones


def analyze_json(drug_data):
    """
    Анализ JSON-данных с подсчетом полей и отсутствующих данных.
    """
    # Подсчет всех возможных полей
    field_counts = {}
    missing_fields = {}

    for drug in drug_data:
        drug_name = drug.get('drug', 'Unknown')
        
        # Подсчет полей
        for field in drug.keys():
            field_counts[field] = field_counts.get(field, 0) + 1
        
        # Проверка отсутствующих полей
        all_possible_fields = {'drug', 'group', 'text', 'text_side_e', 
                                'dosage_form', 'form_release',
                                'contraindications', 'caution'}
        missing = all_possible_fields - set(drug.keys())
        if missing:
            missing_fields[drug_name] = list(missing)

    print("Количество каждого типа поля:")
    for field, count in field_counts.items():
        print(f"{field}: {count}")

    print("\nОтсутствующие поля по препаратам:")
    for drug, fields in missing_fields.items():
        print(f"{drug}: {', '.join(fields)}")

    return field_counts, missing_fields
    
if __name__ == "__main__":

    # drug_data = []

    drug_dict_from_pdf = extract_text_pdf()

    # drug_data.extend(drug_dict_from_pdf)
    
    analyze_json(drug_dict_from_pdf)

    contraind_extractor = ContraindicationExtractor()
    kinetic_extractor = PharmacokineticsExtractor()

    # Извлечение фармакокинетики
    for drug_data in drug_dict_from_pdf:
        text = drug_data.get('text', '')
        dosage_form = drug_data.get('dosage_form', '')
        contraindication = drug_data.get("contraindications", "")
        caution = drug_data.get("caution", "")

        drug_data['pharmacokinetics'] = kinetic_extractor.extract_elimination_periods(text)
        drug_data['protein_binding'] =  kinetic_extractor.extract_protein_binding(text)
        drug_data['filter_dosage'] = extract_dosage_form(dosage_form)
        drug_data['extracted_contraindication'] = contraind_extractor.extract(contraindication)
        drug_data['extracted_caution'] = contraind_extractor.extract(caution)

    # # Извлечение лекарственной формы
    # for drug_data in drug_dict_from_pdf:
    #     extract_dosage_form(drug_data)

    # # Извлечение противопоказаний
    # extractor = ContraindicationExtractor()
    # for drug_data in drug_dict_from_pdf:
    #     contraindication = drug_data.get("contraindications", "")
    #     caution = drug_data.get("caution", "")

    with open(f"{DIR_SAVE}\\extracted_data_all.json", "w", encoding="utf-8") as json_file:
        json.dump(drug_dict_from_pdf, json_file, ensure_ascii=False, indent=4)