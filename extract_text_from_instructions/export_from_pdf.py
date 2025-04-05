import fitz  # PyMuPDF
import re
import json

import os

import sys
sys.path.append("")

from convertors.normalize_text import normalize_text

DIR_LIST_PDF = [
    "data\\Инструкции_ГРЛС_не_вкладыши_не_сканы_1",
    "data\\Инструкции_ГРЛС_не_вкладыши_не_сканы_2"
]

# def extract_text_from_pdf(pdf_path):
#     with fitz.open(pdf_path) as pdf:
#         text = ""
#         for page in pdf:
#             page_text = page.get_text("text")
#             filtered_lines = []
#             for line in page_text.split("\n"):
#                 stripped_line = line.strip()
#                 if not stripped_line.isdigit() and len(stripped_line) >= 3:
#                     filtered_lines.append(stripped_line)
#             text += "\n".join(filtered_lines) + "\n"
#     return text


def extract_text_from_pdf(pdf_path):
    with open("data_jsonl_export\\raw_text.txt", "a", encoding="utf-8") as f:
        with fitz.open(pdf_path) as pdf:
            text = ""
            previous_line = ""
            
            filtered_lines = []
            for page in pdf:
                page_text = page.get_text("text")
                
                f.write(f"{pdf_path}: {page_text}")
                
                for line in page_text.split("\n"):
                    stripped_line = line.strip()
                    if not stripped_line.isdigit() and len(stripped_line) >= 3:
                        if previous_line.endswith("-") and filtered_lines:
                            # Соединяем перенос слова, если есть предыдущая строка
                            filtered_lines[-1] = filtered_lines[-1][:-1] + stripped_line
                        else:
                            filtered_lines.append(stripped_line)
                        previous_line = stripped_line
                
            text += "\n".join(filtered_lines) + "\n"
    
    return text


def extract_sections(name, text):
    sections = {}

    sections["drug"] = name

    # Извлекаем текст после "Фармакотерапевтическая группа" и до "Код АТХ", который начинается с новой строки
    match_pharmacotherapeutic = re.search(
        r'(Фармакотерапевтическая\s*группа|ФАРМАКОТЕРАПЕВТИЧЕСКАЯ\s*ГРУППА)(?:\s*[.:]\s*)?(.*?)(?:\n|\A)(Код АТХ|КОД ATX|Код ATX|КОД АТХ|Код ATX|Код АТX|Код ATХ)',
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
    
    return sections

def extract_pdf(dir_list = DIR_LIST_PDF):

    drug_jsones = []
    
    for directory in list(dir_list):

        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                name = os.path.splitext(filename)[0]
                pdf_path = os.path.join(directory, filename)
                
                extracted_text = extract_text_from_pdf(pdf_path)
                drug_sections = extract_sections(name, extracted_text)
                
                if drug_sections:
                    drug_jsones.append(drug_sections)

    return drug_jsones

def main():
    # drug_jsones = []
    
    # for directory in [DIR_PDF_1, DIR_PDF_2]:
    #     cycle_dir(directory, drug_jsones)

    drug_jsones = extract_pdf()
    
    output_path = os.path.join("data_jsonl_export", "extracted_data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(drug_jsones, json_file, ensure_ascii=False, indent=4)
    
    print("Данные сохранены в", output_path)

if __name__ == "__main__":
    main()
