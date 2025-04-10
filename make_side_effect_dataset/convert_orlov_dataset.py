import json
import os
import logging

import sys
sys.path.append("")

from CustomPymorphy.CustomPymorphy import EnhancedMorphAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Анализатор морфологии
custom_morph = EnhancedMorphAnalyzer()

# Пути к файлам
DIR_INPUT = "make_side_effect_dataset\\data\\side_e_dataset.json"
DIR_OUTPUT = "make_side_effect_dataset\\data\\sef_dataset.json"


def load_dataset(filepath):
    """Загружает JSON-файл с датасетом."""
    if not os.path.exists(filepath):
        logging.error(f"Файл {filepath} не найден")
        raise FileNotFoundError(f"Файл {filepath} не найден")

    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def process_drug_data(dataset):
    """Обрабатывает данные о лекарствах и формирует новую структуру."""
    processed_data = {}

    for drug in dataset:
        if not drug.get("source"):
            continue  # Пропускаем записи без источника

        drug_name_ru = drug["drug_name_ru"].lower()

        # Новая структура данных
        drug_info = {
            "drug_name_en": drug["drug_name_en"],
            "side_e_parts": {},
        }
        side_e_parts = drug.get("side_e_parts")
        if side_e_parts:
            for section, content in side_e_parts.items():
                if isinstance(content, dict):
                    # Структура: { "часто": [...], "редко": [...] }
                    for freq, effects in content.items():
                        for effect in effects:
                            new_effect = custom_morph.lemmatize_string(effect.lower())
                            drug_info["side_e_parts"][new_effect] = freq
                elif isinstance(content, list):
                    # Структура: просто список побочек без подзаголовков
                    for effect in content:
                        new_effect = custom_morph.lemmatize_string(effect.lower())
                        drug_info["side_e_parts"][new_effect] = "Частота неизвестна"
                        # print(f"  - {effect}")
                else:
                    print(f"Неожиданный формат для '{section}': {type(content)}")

        processed_data[drug_name_ru] = drug_info

    return processed_data


def save_dataset(data, filepath):
    """Сохраняет обработанные данные в JSON-файл."""
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    logging.info(f"Обработанные данные сохранены в {filepath}")


if __name__ == "__main__":
    dataset = load_dataset(DIR_INPUT)
    processed_data = process_drug_data(dataset)
    save_dataset(processed_data, DIR_OUTPUT)
