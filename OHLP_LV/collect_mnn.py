import os

def collect_prepare_names(pdf_dir: str, output_file: str = "prepare_names.txt") -> None:
    prepare_names = set()

    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith('.pdf'):
            continue

        # 1. Разделить по первой ")" — получим номер и остаток
        num_prepare, rest = filename.split(')', 1)

        # 2. Убрать пробелы в начале остатка
        rest = rest.strip()

        # Убираем расширение .pdf
        parts = rest.split(' _ ')

        if len(parts) != 3:
            print(f"Пропущен файл с неверным форматом: {filename}")
            continue

        prepare_name = parts[0].strip()
        prepare_names.add(prepare_name)

    # Запись в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in sorted(prepare_names):
            f.write(name + '\n')

    print(f"Собрано {len(prepare_names)} уникальных наименований. Сохранено в {output_file}")

# Пример использования:
collect_prepare_names("OHLP_LV\\data\\ОХЛП_all", "OHLP_LV\\data\\prepare_names.txt")