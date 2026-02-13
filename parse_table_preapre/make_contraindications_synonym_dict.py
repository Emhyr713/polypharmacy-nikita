import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set
import logging

sys.path.append("")
from create_graph.SemanticEmbeddingProcessor import SemanticEmbeddingProcessor

def load_json_data(file_path: Path) -> List[dict]:
    """Загружает JSON и возвращает список объектов."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [data] if isinstance(data, dict) else data


def load_text_lines(file_path: Path) -> List[str]:
    """Загружает строки из текстового файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f if line.strip()]


def extract_unique_contraindications(data: List[dict]) -> List[str]:
    """Извлекает уникальные противопоказания из JSON-данных."""
    unique = set()
    for item in data:
        contraindications = item.get("extracted_contraindication", [])
        if isinstance(contraindications, list):
            for value in contraindications:
                if isinstance(value, str) and value.strip():
                    unique.add(value.strip())
    return sorted(unique)


def normalize_terms(raw_terms: List[str], 
                   standard_terms: List[str],
                   model_path: Path,
                   threshold: float = 0.97,
                   top_n: int = 1) -> Dict[str, List[str]]:
    """Нормализует исходные термины к эталонным с помощью семантической модели."""
    
    if not raw_terms or not standard_terms:
        return {}
    
    processor = SemanticEmbeddingProcessor(str(model_path))
    matches = processor.find_similar_terms(raw_terms, standard_terms, threshold, top_n)
    
    result = defaultdict(list)
    for original, match_list in matches.items():
        if match_list:
            result[match_list[0]['term']].append(original)
    
    return dict(result)


def main():
    # Конфигурация
    paths = {
        'input': Path("parse_table_preapre/data/ЛС 02.2026.json"),
        'standards': Path("parse_table_preapre/data/standart_contraindications.txt"),
        'output': Path("parse_table_preapre/data/dict_synonym_contraindications.json"),
        'model': Path("train_synonim_model/data/synonym-model_4")
    }

    # 1. Загрузка и извлечение данных
    print("Загрузка JSON...")
    json_data = load_json_data(paths['input'])
    
    print("Извлечение уникальных противопоказаний...")
    raw_terms = extract_unique_contraindications(json_data)
    print(f"Найдено {len(raw_terms)} уникальных терминов")
    
    # 2. Загрузка эталонов
    standard_terms = load_text_lines(paths['standards'])
    print(f"Загружено {len(standard_terms)} эталонных терминов")
    
    # 3. Нормализация
    print("Нормализация терминов...")
    normalized = normalize_terms(raw_terms, standard_terms, paths['model'])
    
    # 4. Сохранение
    paths['output'].parent.mkdir(parents=True, exist_ok=True)
    with open(paths['output'], 'w', encoding='utf-8') as f:
        json.dump(normalized, f, ensure_ascii=False, indent=4)
    
    print(f"Результат: {paths['output']}")
        
if __name__ == "__main__":
    main()