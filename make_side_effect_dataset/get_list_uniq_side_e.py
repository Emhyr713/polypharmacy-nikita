 
import json

def get_list_uniq_side_e(json_data):
    uniq_set = set()

    for drug, content in json_data.items():
        for side_e in content['side_e_parts']:
            uniq_set.add(side_e)

    return list(uniq_set)

if __name__ == "__main__":
    filename = "make_side_effect_dataset\\data\\sef_dataset.json"
    with open(filename, "r", encoding="utf-8") as file:
        json_dataset = json.load(file)

    uniq_list = get_list_uniq_side_e(json_dataset)

    filepath = "make_side_effect_dataset\\data\\sef_uniq_list.txt"
    with open(filepath, "w", encoding="utf-8") as file:
        file.write("\n".join(uniq_list))