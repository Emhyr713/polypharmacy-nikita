import json
import os

# Имя входного текстового файла
INPUT_TXT_FILENAME = "make_side_effect_dataset\\data\\all_side_e.txt"  # ← укажи свой путь

# Имя выходного JSON-файла
OUTPUT_JSON_FILENAME = "visualization_embedding\\data\\side_effects_dataset.json"

# Создаем директорию, если её нет
os.makedirs(os.path.dirname(OUTPUT_JSON_FILENAME), exist_ok=True)

# Чтение и обработка строк
side_effects = []

with open(INPUT_TXT_FILENAME, 'r', encoding='utf-8') as f:
    for line in f:
        stripped = line.strip()
        if stripped:  # пропускаем пустые строки
            side_effects.append(stripped)

# Убираем дубликаты, сохраняя порядок
side_effects = list(dict.fromkeys(side_effects))

# Формируем структуру: [[строка, "side_effect"], ...]
side_effects_json = [[effect, "side_effect"] for effect in side_effects]

# Сохраняем в JSON
with open(OUTPUT_JSON_FILENAME, 'w', encoding='utf-8') as f:
    json.dump(side_effects_json, f, ensure_ascii=False, indent=2)

print(f"✅ Извлечено {len(side_effects)} побочных эффектов → сохранено в {OUTPUT_JSON_FILENAME}")