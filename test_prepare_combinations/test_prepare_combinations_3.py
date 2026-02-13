import itertools
import os
import requests
import time
from tqdm import tqdm
import csv
import pandas as pd

from fortran_calc import FortranCalculator

valid_ids = range(1, 93)
WORK_DIR = "test_prepare_combinations"
os.makedirs(WORK_DIR, exist_ok=True)

base_url = "http://158.160.151.7/api/polifarmakoterapiya-fortran/"
headers = {"Accept": "application/json"}

FILENAME_SIDE_E = "test_prepare_combinations\\data\\side_e_ids.csv"
FILENAME_DRUG = "test_prepare_combinations\\data\\drug_ids.csv"

def load_item2id(filename):
    # Загрузка соответствия название ЛС ↔ ID
    item2id = {}
    id2item = {}
    with open(filename, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            item2id[row["item"].strip().lower()] = int(row["id"])
            id2item[int(row["id"])] = row["item"].strip().lower()
    return item2id, id2item
    
side_e2id, id2side_e = load_item2id(FILENAME_SIDE_E)
drug2id, id2drug = load_item2id(FILENAME_DRUG)

# n_side_e = len(side_e2id)
# n_drug = len(drug2id)

# Чтение CSV-файла (предполагается, что разделитель запятая)
df = pd.read_csv("test_prepare_combinations\\data\\weights.csv", header=None, delimiter=';')

# Преобразование DataFrame в numpy array, затем в одномерный список
weight_string = df.values.flatten().tolist()

print(weight_string)

calc = FortranCalculator(id2drug, id2side_e, weight_string)

# Загрузка таблицы с парами ЛС
df = pd.read_csv(f"{WORK_DIR}\\data\\compatible_2.csv")
print("df:", df)

# Приведение названий ЛС к нижнему регистру
df["ЛС1"] = df["ЛС1"].str.lower().str.strip()
df["ЛС2"] = df["ЛС2"].str.lower().str.strip()

# Замена названий на ID
df["id1"] = df["ЛС1"].map(drug2id)
df["id2"] = df["ЛС2"].map(drug2id)

# Отфильтровываем пары, в которых один из ID не найден
df = df.dropna(subset=["id1", "id2"]).astype({"id1": int, "id2": int})
print("df.dropna:", df)


# Загрузка второй таблицы — с уже проверенными или исключёнными парами
df_existing_pairs = pd.read_csv("test_prepare_combinations\\data\\incompatible_or_caution_2.csv")

# Приведение к нижнему регистру и сопоставление с ID
df_existing_pairs["ЛС1"] = df_existing_pairs["ЛС1"].str.lower().str.strip()
df_existing_pairs["ЛС2"] = df_existing_pairs["ЛС2"].str.lower().str.strip()
df_existing_pairs["id1"] = df_existing_pairs["ЛС1"].map(drug2id)
df_existing_pairs["id2"] = df_existing_pairs["ЛС2"].map(drug2id)

# Создаём множество запрещённых пар как frozenset для симметричной проверки (A,B) ≡ (B,A)
excluded_pairs = set(
    frozenset((row["id1"], row["id2"]))
    for _, row in df_existing_pairs.dropna(subset=["id1", "id2"]).iterrows()
)


# Создание CSV-файлов
with (
    open(f"{WORK_DIR}/data/incompatible_or_caution_3.csv", mode="w", newline="", encoding="utf-8") as inc_file,
    open(f"{WORK_DIR}/data/compatible_triples_3.csv", mode="w", newline="", encoding="utf-8") as comp_file
):
    inc_writer = csv.writer(inc_file)
    comp_writer = csv.writer(comp_file)

    headers_row = ["ЛС1", "ЛС2", "ЛС3", "Тип совместимости", "Первый побочный эффект", "Ранг"]
    inc_writer.writerow(headers_row)
    comp_writer.writerow(headers_row)

    all_triples = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        id1, id2 = row["id1"], row["id2"]

        used_ids = {id1, id2}
        possible_third = sorted(set(valid_ids) - used_ids)
        # print("possible_third:", possible_third)

        for id3 in possible_third:
            triple = sorted([id1, id2, id3])
            triple_set = set(triple)

            # Все пары внутри тройки
            pair1 = frozenset((id1, id2))
            pair2 = frozenset((id1, id3))
            pair3 = frozenset((id2, id3))

            if triple in all_triples:
                continue  # Исключаем дублирование троек

            if pair1 in excluded_pairs or pair2 in excluded_pairs or pair3 in excluded_pairs:
                # print("пропуск")
                continue  # Пропустить тройку, если хотя бы одна пара запрещена

            all_triples.append(triple)

            # drugs_param = f"[{triple[0]},+{triple[1]},+{triple[2]}]"
            # url = f"{base_url}?drugs={drugs_param}&humanData=0"

            try:
                # response = requests.get(url, headers=headers, timeout=15)
                # response.raise_for_status()
                # data = response.json()

                data = calc.calculate(list(triple))

                drugs = data["drugs"]
                compatibility = data["сompatibility_fortran"].lower()
                side_effects = data.get("side_effects", None)

                se_name = "нет данных"
                rank = "нет данных"

                if side_effects:
                    matched_effects = next(
                        (block["effects"] for block in side_effects if block.get("сompatibility", "").lower() == compatibility),
                        []
                    )

                    if matched_effects:
                        first_effect = matched_effects[0]
                        se_name = first_effect.get("se_name", "нет данных")
                        rank = first_effect.get("rank", "нет данных")

                output_row = [*drugs, compatibility, se_name, rank]

                if compatibility in ("incompatible", "caution", "banned"):
                    inc_writer.writerow(output_row)
                elif compatibility == "compatible":
                    comp_writer.writerow(output_row)
                else:
                    print(f"Неизвестный тип совместимости: {compatibility} для тройки {triple}")

            except requests.exceptions.RequestException as e:
                print(f"Ошибка сети при обработке тройки {triple}: {e}")
            except ValueError as e:
                print(f"Ошибка декодирования JSON для тройки {triple}: {e}")
            # except KeyError as e:
            #     print(f"Отсутствует ключ в данных для тройки {triple}: {e}")
            # except Exception as e:
            #     print(f"Неожиданная ошибка при обработке тройки {triple}: {e}")

            # time.sleep(0.1)