import sqlite3
import json

# Подключаемся к файлу базы данных
conn = sqlite3.connect("data\\graph_data_yed_processed\\graph.db")
cursor = conn.cursor()

# Выполняем SQL-запрос
cursor.execute("SELECT label FROM entities")

# Получаем все значения в списке
column_values = [row[0] for row in cursor.fetchall()]

# Закрываем соединение
conn.close()

ent_dict = {label:[label] for label in column_values}

# Выводим результат
print(column_values)

with open("data_jsonl_export\\seq2seq_dataset.json", "w", encoding="utf-8") as json_file:
    json.dump(ent_dict, json_file, ensure_ascii=False, indent=4)
