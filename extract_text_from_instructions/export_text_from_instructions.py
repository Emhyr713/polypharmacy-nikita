import fitz  # PyMuPDF
import re
import json
import os
import sys
sys.path.append("")

# from convertors.normalize_text import normalize_text
from utils.extract_text_from_pdf import extract_text_from_pdf

DIR_LIST_PDF = [
    "data\\Инструкции_ГРЛС_1",
    "data\\Инструкции_ГРЛС_2"
]

JSON_LIST_SCANS = [
    # "data\\Инструкции_ГРЛС_1_scan\\Инструкции_ГРЛС_1_scan.json",
    # "data\\Инструкции_ГРЛС_2_scan\\Инструкции_ГРЛС_2_scan.json",
]

DIR_SAVE = "extract_text_from_instructions\\data"

def extract_sections(name, text):
    sections = {}

    sections["drug"] = name

    match_pharmacotherapeutic = re.search(
    r'(Фармакотерапевтическая\s*группа|ФАРМАКОТЕРАПЕВТИЧЕСКАЯ\s*ГРУППА)(?:\s*[.:]\s*)?(.*?)(?=\s*[‚\-–—]*\s*[КK]\S*?\s+[AА][ТT][ХX])',
    text,
    re.DOTALL
    )

    if match_pharmacotherapeutic:
        text_groups = match_pharmacotherapeutic.group(2).strip()  
        text_groups = re.sub(r'\n', ' ', text_groups)  
        sections["group"] = text_groups
    
    # Извлекаем текст от "Фармакодинамика" до "Показания к применению", учитывая, что "Показания к применению" должны быть на новой строке
    match_pharmacodynamics = re.search(
        r'(Фармакологические\s*свойства|Фармакологическое\s*действие|ФАРМАКОЛОГИЧЕСКИЕ\s*СВОЙСТВА)(?:\s*[.:]\s*)?(.*?)(?:\n|\A)(Показания к применению|Показания к медицинскому применению|ПОКАЗАНИЯ К ПРИМЕНЕНИЮ)',
        text,
        re.DOTALL
    )
    
    if match_pharmacodynamics:
        text_main = match_pharmacodynamics.group(2).strip()
        text_main = re.sub(r'\n', ' ', text_main)
        text_main = re.sub(r'\s+', ' ', text_main)
        sections["text"] = text_main


    # Извлекаем текст от "Побочные действия" до "Передозировка"
    match_side_e = re.search(
        r'(\nПобочн(?:ое|ые)\s*действ(?:ие|ия)|\nПОБОЧН(?:ОЕ|ЫЕ)\s*ДЕЙСТВ(?:ИЕ|ИЯ))(?:\s*[.:]\s*)?(.*?)(?:\n|\A)(Передозировка|ПЕРЕДОЗИРОВКА)',
        text,
        re.DOTALL
    )
    
    if match_side_e:
        text_main = match_side_e.group(2).strip()
        text_main = re.sub(r'\n', ' ', text_main)
        text_main = re.sub(r'\s+', ' ', text_main)
        sections["text_side_e"] = text_main

    return sections

def extract_text_pdf(dir_list = DIR_LIST_PDF):

    drug_jsones = []
    
    for directory in list(dir_list):

        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                name = os.path.splitext(filename)[0]
                name = name.replace("_OCR", "")
                pdf_path = os.path.join(directory, filename)
                
                extracted_text = extract_text_from_pdf(pdf_path)
                drug_sections = extract_sections(name, extracted_text)
                
                if drug_sections:
                    drug_jsones.append(drug_sections)

    return drug_jsones

def extract_text_scans(json_list = JSON_LIST_SCANS):
    drug_jsones = []
    for json_file in list(json_list):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for insructions in data:
            drug_sections = extract_sections(insructions["drug"], insructions["text"])
            if drug_sections:
                drug_jsones.append(drug_sections)
    return drug_jsones

def main():

    drug_data = []

    drug_dict_from_pdf = extract_text_pdf()
    drug_dict_from_scans = extract_text_scans()

    drug_data.extend(drug_dict_from_pdf)
    drug_data.extend(drug_dict_from_scans)
    
    with open(f"{DIR_SAVE}\\extracted_data_all.json", "w", encoding="utf-8") as json_file:
        json.dump(drug_data, json_file, ensure_ascii=False, indent=4)

    with open(f"{DIR_SAVE}\\extracted_data_pdf.json", "w", encoding="utf-8") as json_file:
        json.dump(drug_dict_from_pdf, json_file, ensure_ascii=False, indent=4)

    with open(f"{DIR_SAVE}\\extracted_data_scans.json", "w", encoding="utf-8") as json_file:
        json.dump(drug_dict_from_scans, json_file, ensure_ascii=False, indent=4)
    
    
if __name__ == "__main__":
    main()
