import pandas as pd
import json
import re

def clean_string(value):
    """Очищает строку: удаляет символы новой строки, лишние пробелы, приводит к нижнему регистру"""
    if pd.isna(value):
        return ""
    # Преобразуем в строку, удаляем \n и другие пробельные символы, приводим к нижнему регистру
    cleaned = str(value).strip().lower()
    # Заменяем все последовательности пробельных символов (включая \n, \t, \r) на один пробел
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

def parse_string_to_list(value):
    """Преобразует строку с разделителем ';' в список, очищая и приводя к нижнему регистру"""
    if pd.isna(value) or value == '-' or value == '':
        return []
    
    # Очищаем всю строку
    cleaned_value = clean_string(value)
    
    # Разделяем по точке с запятой, фильтруем пустые
    items = [item.strip() for item in cleaned_value.split(';') if item.strip()]
    return items

def convert_flag(value):
    """Преобразует флаги +/- в булевы значения"""
    if str(value).strip() == '+':
        return True
    elif str(value).strip() == '-':
        return False
    return None

def main():
    # Чтение Excel файла
    file_path = 'parse_table_preapre\\data\\ЛС 02.2026.xlsx'  # Укажите путь к вашему файлу
    
    try:
        # Читаем лист "ЛС2"
        df = pd.read_excel(file_path, sheet_name='ЛС2')
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return
    
    # Проверяем наличие необходимых колонок
    required_columns = ['Название препарата', 'Группа', 'Запрещенные ЛС', 
                       'Запрещенные группы', 'С осторожностью', 'С осторожностью группы',
                       'Беременность +/-', 'Г.В.+/-', 'до 18 +/-', 'после 65 +/-', 
                       'Противопоказания']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Отсутствуют необходимые колонки: {missing_columns}")
        print(f"Доступные колонки: {list(df.columns)}")
        return
    
    # Список для хранения результатов
    result = []
    
    # Обрабатываем каждую строку
    for _, row in df.iterrows():
        # Пропускаем пустые строки (где нет названия препарата)
        if pd.isna(row['Название препарата']):
            continue
            
        # Создаем запись для препарата
        drug_entry = {
            "drug": clean_string(row['Название препарата']),
            "group": clean_string(row['Группа']),
            "banned_drugs": parse_string_to_list(row['Запрещенные ЛС']),
            "banned_groups": parse_string_to_list(row['Запрещенные группы']),
            "warning_drugs": parse_string_to_list(row['С осторожностью']),
            "warning_groups": parse_string_to_list(row['С осторожностью группы']),
            "banned_pregnancy": convert_flag(row['Беременность +/-']),
            "banned_breastfeeding": convert_flag(row['Г.В.+/-']),
            "banned_under_18": convert_flag(row['до 18 +/-']),
            "banned_after_18": convert_flag(row['после 65 +/-']),
            "extracted_contraindication": parse_string_to_list(row['Противопоказания'])
        }
        
        result.append(drug_entry)
    
    # Сохраняем в JSON файл
    output_file = 'parse_table_preapre\\data\\ЛС 02.2026.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Данные успешно сохранены в файл: {output_file}")
    print(f"Обработано записей: {len(result)}")
    
    # Выводим пример первой записи для проверки
    if result:
        print("\nПример первой записи:")
        print(json.dumps(result[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()