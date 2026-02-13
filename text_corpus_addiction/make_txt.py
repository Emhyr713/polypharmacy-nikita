import json

# Имена файлов
JSON_FILENAME = "text_corpus_addiction/data/rlsnet_texts.json"
TXT_OUTPUT_FILENAME = "text_corpus_addiction/data/rlsnet_texts_from_json.txt"

# Читаем JSON
with open(JSON_FILENAME, 'r', encoding="utf-8") as f:
    data = json.load(f)

# Готовим строки для TXT
txt_lines = []

for item in data:
    # Каждый элемент — словарь с одним ключом (название препарата)
    drug_name = list(item.keys())[0]
    text_list = item[drug_name]  # список строк или None

    if text_list is not None:
        # Объединяем список строк в один текст
        full_text = "\n".join(text_list)
        txt_lines.append(full_text)

# Сохраняем в TXT
with open(TXT_OUTPUT_FILENAME, 'w', encoding="utf-8") as f:
    f.write("\n".join(txt_lines))

print(f"✅ Текст успешно сохранён в {TXT_OUTPUT_FILENAME}")