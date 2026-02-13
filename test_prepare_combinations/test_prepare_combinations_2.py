import requests
import itertools
import time
import csv
from tqdm import tqdm
import os

valid_ids = range(1, 93)  # ID препаратов от 1 до 92

WORK_DIR = "test_prepare_combinations"
os.makedirs(WORK_DIR, exist_ok=True)  # Создаем директорию, если ее нет

base_url = "http://158.160.151.7/api/polifarmakoterapiya-fortran/"
headers = {"Accept": "application/json"}

# Открываем два файла для записи
with (
    open(f"{WORK_DIR}/incompatible_or_caution.csv", mode="w", newline="", encoding="utf-8") as inc_file,
    open(f"{WORK_DIR}/compatible.csv", mode="w", newline="", encoding="utf-8") as comp_file
):
    # Создаем writer'ы для каждого файла
    inc_writer = csv.writer(inc_file)
    comp_writer = csv.writer(comp_file)

    # Заголовки для обоих файлов
    headers_row = ["ЛС1", "ЛС2", "Тип совместимости", "Первый побочный эффект", "Ранг"]
    inc_writer.writerow(headers_row)
    comp_writer.writerow(headers_row)

    total_combinations = len(list(itertools.combinations(valid_ids, 2)))
    
    for id1, id2 in tqdm(itertools.combinations(valid_ids, 2), total=total_combinations):
        drugs_param = f"[{id1},+{id2}]"
        url = f"{base_url}?drugs={drugs_param}&humanData=0"

        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            drugs = data["data"]["drugs"]
            compatibility = data["data"]["сompatibility_fortran"].lower()  # Приводим к нижнему регистру
            side_effects = data["data"].get("side_effects", None)  # Исправлено: убраны лишние квадратные скобки

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

            # row = drugs.append(compatibility, se_name, rank)
            row = [*drugs, compatibility, se_name, rank]

            # Записываем в соответствующий файл в зависимости от типа совместимости
            if compatibility in ("incompatible", "caution", "banned"):
                inc_writer.writerow(row)
            elif compatibility == "compatible":
                comp_writer.writerow(row)
            else:
                print(f"Неизвестный тип совместимости: {compatibility} для пары {id1},{id2}")

        except requests.exceptions.RequestException as e:
            print(f"Ошибка сети при обработке пары {id1},{id2}: {e}")
        except ValueError as e:
            print(f"Ошибка декодирования JSON для пары {id1},{id2}: {e}")
        except KeyError as e:
            print(f"Отсутствует ожидаемый ключ в данных для пары {id1},{id2}: {e}")
        except Exception as e:
            print(f"Неожиданная ошибка при обработке пары {id1},{id2}: {e}")
        
        time.sleep(0.1)  # Задержка между запросами