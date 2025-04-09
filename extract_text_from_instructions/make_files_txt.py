import json
import os

DIR_SAVE = "extract_text_from_instructions\\data\\Инструкции_txt"
FILENAME = "extract_text_from_instructions\\data\\extracted_data_all.json"

def save_drugs_from_json(json_file):
    # Читаем JSON-файл
    with open(json_file, 'r', encoding='utf-8') as file:
        drugs = json.load(file)

    # Создаем файлы с названиями препаратов
    for drug in drugs:
        name = drug.get("drug", "Без имени")
        text = drug.get("text", "")
        
        with open(f"{DIR_SAVE}\\{name}.txt", 'w', encoding='utf-8') as f:
            f.write(text) 

    return len(drugs)

if __name__ == "__main__":

    count_saved_drugs = save_drugs_from_json(FILENAME)
    print(f"Создано {count_saved_drugs} файлов")

