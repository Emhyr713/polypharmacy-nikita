import json
import os

# Имя входного файла
INPUT_FILENAME = "extract_data_from_instructions\\data\\extracted_data_all.json"

# Имена выходных файлов
CONTRAINDICATIONS_FILE = "visualization_embedding\\data\\contraindications_dataset.json"
CAUTIONS_FILE = "visualization_embedding\\data\\cautions_dataset.json"

# Создаем директорию, если её нет
os.makedirs(os.path.dirname(CONTRAINDICATIONS_FILE), exist_ok=True)

# Списки для сбора данных
all_contraindications = []
all_cautions = []

# Чтение и обработка
with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
    data = json.load(f)  # предполагается, что это список словарей

for item in data:
    # Извлекаем противопоказания
    contraindications = item.get("extracted_contraindication", [])
    if isinstance(contraindications, list):
        all_contraindications.extend([
            line.strip() for line in contraindications if isinstance(line, str)
        ])

    # Извлекаем "с осторожностью"
    cautions = item.get("extracted_caution", [])
    if isinstance(cautions, list):
        all_cautions.extend([
            line.strip() for line in cautions if isinstance(line, str)
        ])

# Убираем дубликаты, сохраняя порядок
all_contraindications = list(dict.fromkeys(all_contraindications))
all_cautions = list(dict.fromkeys(all_cautions))

# Формируем структуру для JSON
contraindications_json = [[item, "contraindication"] for item in all_contraindications]
cautions_json = [[item, "caution"] for item in all_cautions]

# Сохраняем противопоказания в JSON
with open(CONTRAINDICATIONS_FILE, 'w', encoding='utf-8') as f:
    json.dump(contraindications_json, f, ensure_ascii=False, indent=2)

# Сохраняем "с осторожностью" в JSON
with open(CAUTIONS_FILE, 'w', encoding='utf-8') as f:
    json.dump(cautions_json, f, ensure_ascii=False, indent=2)

print(f"✅ Извлечено {len(all_contraindications)} противопоказаний → сохранено в {CONTRAINDICATIONS_FILE}")
print(f"✅ Извлечено {len(all_cautions)} предостережений → сохранено в {CAUTIONS_FILE}")


