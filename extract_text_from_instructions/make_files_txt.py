import json
import os

FILE_DIR = "extract_text_from_instructions\\data"

def save_drugs_from_json(json_file):
    
    # Читаем JSON-файл
    with open(f"{FILE_DIR}\\{json_file}", 'r', encoding='utf-8') as file:
        drugs = json.load(file)
    
    # Создаем файлы с названиями препаратов
    for drug in drugs:
        name = drug.get("drug", "Без имени")
        text = drug.get("text", "")
        
        with open(f"{FILE_DIR}\\Инструкции_txt\\{name}.txt", 'w', encoding='utf-8') as f:
            f.write(text)
    
    print(f"Создано {len(drugs)} файлов")


if __name__ == "__main__":
    for filename in os.listdir(FILE_DIR):
        if filename.endswith(".json"):
            save_drugs_from_json(filename)