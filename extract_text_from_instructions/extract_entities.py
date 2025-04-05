import sqlite3
import json

import sys
sys.path.append("")

from convertors.BIO2umf import BIO2umf_d


filename = "data\\data_6.jsonl"

def save_in_sqltable(table_name, ent_list):
    # Создание и подключение к базе данных (в случае её отсутствия будет создана)
    conn = sqlite3.connect(table_name)
    cursor = conn.cursor()

    # Удаление таблиц, если они существуют
    cursor.execute("DROP TABLE IF EXISTS entities")
    # cursor.execute("DROP TABLE IF EXISTS edges")
    
    # Создание таблицы для узлов
    cursor.execute('''
    CREATE TABLE entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        tag TEXT
    )
    ''')

    # Функция для добавления узлов в таблицу
    for label, tag in ent_list:
        cursor.execute("INSERT INTO entities (label, tag) VALUES (?, ?)", (label, None))
    
    # Сохраняем изменения и закрываем соединение
    conn.commit()

    # Закрытие соединения
    conn.close()

if __name__ == "__main__":

    with open("data_bio\\data_bio_6.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

    ent_list_all = BIO2umf_d(data)
    ent_list = [ent for ent in ent_list_all if ent[1]]
    # for ent, tag in ent_list:
    #     if tag:
    #         print(ent, tag)
    # print(len(ent_list))

    dir_graph_processed = "data\\graph_data_yed_processed"
    save_in_sqltable(f"{dir_graph_processed}\\graph.db", ent_list)
    



