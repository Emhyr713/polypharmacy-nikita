import os
import shutil
from pathlib import Path
import hashlib
from tqdm import tqdm

def file_hash(filepath):
    """Возвращает SHA-256 хеш содержимого файла."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def merge_and_renumber_folders(folder1: str, folder2: str, output_folder: str):
    """
    Объединяет PDF-файлы из двух папок, удаляет дубликаты по содержимому,
    переименовывает с единой нумерацией и сохраняет уникальные названия в TXT.
    """
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Собираем все PDF-файлы из обеих папок
    all_files = list(folder1_path.glob("*.pdf")) + list(folder2_path.glob("*.pdf"))

    seen_hashes = set()
    unique_files = []

    # Фильтруем дубликаты по хешу содержимого — с прогресс-баром
    print("Проверка дубликатов...")
    for file_path in tqdm(sorted(all_files, key=lambda p: p.name), desc="Хеширование"):
        h = file_hash(file_path)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_files.append(file_path)

    # Сортируем уникальные файлы по имени
    unique_files.sort(key=lambda p: p.name)

    unique_names = []

    # Копируем и переименовываем — с прогресс-баром
    print("Копирование и переименование файлов...")
    for idx, file_path in enumerate(tqdm(unique_files, desc="Обработка файлов"), start=1):
        original_name = file_path.stem

        # Убираем начальную нумерацию вида "1) ", "12) " и т.п.
        if ") " in original_name:
            clean_name = original_name.split(") ", 1)[1]
        else:
            clean_name = original_name

        unique_names.append(clean_name)

        new_name = f"{idx}) {clean_name}.pdf"
        new_path = output_path / new_name
        shutil.copy2(file_path, new_path)

    # Сохраняем уникальные названия в TXT-файл
    names_txt_path = output_path / "unique_names.txt"
    with open(names_txt_path, "w", encoding="utf-8") as f:
        for name in unique_names:
            f.write(name + "\n")

    print(f"\n✅ Объединено {len(unique_files)} уникальных файлов в папку '{output_folder}'.")
    print(f"📄 Список уникальных названий сохранён в '{names_txt_path}'.")

# Пример использования:
if __name__ == "__main__":
    DIR = "OHLP_LV\\data"
    merge_and_renumber_folders(f"{DIR}\\ОХЛП_1", f"{DIR}\\ОХЛП_2", f"{DIR}\\ОХЛП_all")