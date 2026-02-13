from itertools import combinations
from collections import defaultdict
import json



def synonim_list2dataset(data):
    clusters = {}
    for i, (key, value_list) in enumerate(data.items()):
        cluster_name = f"cluster_{i}"

        # Используем множество для уникальности
        unique_terms = {key}                # начинаем с ключа
        unique_terms.update(value_list)     # добавляем все значения из списка

        # Преобразуем в список пар [термин, метка] — порядок может быть любым
        labels = [[term, 1] for term in unique_terms]

        clusters[cluster_name] = {
            "labels": labels
        }
    return {"clusters": clusters}

if __name__ == "__main__":

    # SYNONYM_FILENAME_IN = "data\\dictonary_synonims_simple.json"
    # SYNONYM_FILENAME_OUT = "train_synonim_model\\data\\clusters_synosym_dict.json"

    SYNONYM_FILENAME_IN = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"
    SYNONYM_FILENAME_OUT = "train_synonim_model\\data\\clusters_side_e_synonim_dict_all.json"
    
    with open(SYNONYM_FILENAME_IN, "r", encoding="utf-8") as file:
        abbrev_dataset = json.load(file)[0]

    with open(SYNONYM_FILENAME_OUT, "w", encoding="utf-8") as file:
        json.dump(synonim_list2dataset(abbrev_dataset), file, ensure_ascii=False, indent=4)

    # SIDE_E_FILENAME_IN = "make_side_effect_dataset\\data\\side_e_synonim_dict_all.json"
    # SIDE_E_FILENAME_OUT ="train_synonim_model\\data\\clusters_side_e_dict.json"
    
    # with open(SIDE_E_FILENAME_IN, "r", encoding="utf-8") as file:
    #     side_e_filename = json.load(file)

    # side_e_dataset = defaultdict(list)
    # for d in side_e_filename:
    #     for key, values in d.items():
    #         side_e_dataset[key].extend(values)

    # with open(SIDE_E_FILENAME_OUT, "w", encoding="utf-8") as file:
    #     json.dump(synonim_list2dataset(side_e_dataset), file, ensure_ascii=False, indent=4)

    


