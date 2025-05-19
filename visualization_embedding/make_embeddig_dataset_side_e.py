
import json
import sys
sys.path.append("")


if __name__ == "__main__":

    SIDE_E_DATASET = "make_side_effect_dataset\\data\\side_e_dataset.json"
    with open(SIDE_E_DATASET, "r", encoding = "utf-8") as file:
        dataset = json.load(file)

    DICT_CLASS_SIDE_E = "make_side_effect_dataset\\data\\dict_class_side_e.json"
    with open(DICT_CLASS_SIDE_E, "r", encoding = "utf-8") as file:
        dict_class = json.load(file)
    # Инвертируем словарь, разворачивая списки
    map_dict_class  = {
        item: key 
        for key, items in dict_class.items() 
        for item in items
    }

    text_corpus = ""
    result = {}
    for drug in dataset:
        if drug.get("source"):
            side_e_parts = drug["side_e_parts"]
            if drug.get("text"):
                text_corpus += drug["text"]
            for section, content in side_e_parts.items():
                if isinstance(content, dict):
                    for freq, effects in content.items():
                        for sise_e in effects:
                            result[sise_e] = {
                                "section": map_dict_class.get(section, section),  # Используем map_dict_class или исходный section
                                "embedding": None
                            }
                elif isinstance(content, list):
                    for sise_e in content:
                        result[sise_e] = {
                            "section": map_dict_class.get(section, section),  # Используем map_dict_class или исходный section
                            "embedding": None
                        }


    # Пример правильного перебора для списка словарей
    unique_sections = set()
    for item in result:
        side_e = result[item]
        section = side_e["section"]
        unique_sections.add(section)

    print(len(unique_sections))
    print(unique_sections)

    FILE_SAVE = "visualization_embedding\\data\\embedding_blank.json"
    with open(FILE_SAVE, 'w', encoding='utf-8') as f:
        json.dump({"text":text_corpus,"words":result}, f, indent=4, ensure_ascii=False)
