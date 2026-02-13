import re
import csv
import json
import sys
sys.path.append("")

from utils.fetch_url_page import fetch_page_with_retries
from utils.normalize_text import normalize_text
from utils.split_sent_razdel import split_format_text

def clear_text(text):
    text = text.replace("\r\n", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("ё", "е")
    text = re.sub(r"\(см\.[^)]*\)", "", text)
    # text = remove_brackets(text)
    text = text.replace("*", "")
    text = text.replace("§", "")
    text = re.sub(r'\s*/\s*', '/', text)
    text = re.sub(r'\s*:\s*', ':', text)
    text = re.sub(r',\s*$', '', text)
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Удаление строк короче 5 символов (не считая пробелы)
    lines = text.split('\n')
    lines = [line for line in lines if len(line.strip()) >= 5]
    text = '\n'.join(lines)
    return text

def truncate_after_keywords(text, stop_keywords):
    """
    Обрезает текст после точного совпадения с одной из фраз в списке.
    Совпадение чувствительно к порядку слов, но нечувствительно к регистру.
    """
    import re

    earliest_cut = len(text)
    for keyword in stop_keywords:
        # Ищем точное вхождение всей фразы
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        match = pattern.search(text)
        if match and match.start() < earliest_cut:
            earliest_cut = match.start()

    return text[:earliest_cut].strip()

def get_text_from_rlsnet(url):
    """Извлекает текст о побочных эффектах с сайта rlsnet.ru."""
    # url = "https://www.rlsnet.ru" + url_part
    soup = fetch_page_with_retries(url, max_retries = 8)

    if not soup:
        return None

    # Удаляем теги, которые точно не нужны
    for tag in soup.find_all(['sup', 'script', 'style']):
        tag.decompose()

    # Находим секцию с побочными эффектами (проверь ID, может быть другим)
    section = soup.find('h2', id=lambda x: x in ['pharmakologiya'])
    if not section:
        print(f"Не найдена секция с фармакологией эффектами: {url}")
        return None
    
    target_div = None
    for div in section.find_next_siblings('div'):
        if ('style' in div.attrs and 'overflow-wrap: break-word' in div['style']) or \
           ('class' in div.attrs and 'text-break' in div['class']):
            target_div = div
            break

    if not target_div:
        print(f"Не найден div с нужным стилем: {url}")
        return None

    pharmacy_text = clear_text(target_div.get_text(separator=' ', strip=True))
    pharmacy_text = normalize_text(pharmacy_text)

    # Стоп-слова для обрезки текста
    stop_keywords = ['Клинические исследования',
                     'Доклиническая токсикология',
                     'Канцерогенность, мутагенность, влияние на фертильность',
                     'Особые группы пациентов',

                     ]
    pharmacy_text = truncate_after_keywords(pharmacy_text, stop_keywords)

    pharmacy_text_list = split_format_text(pharmacy_text)

    return pharmacy_text_list

if __name__ == "__main__":
    CSV_FILENAME = "text_corpus_addiction/data/rlsnet_links.csv"
    OUTPUT_FILENAME = "text_corpus_addiction/data/rlsnet_texts.json"
    TXT_OUTPUT_FILENAME = "text_corpus_addiction/data/rlsnet_texts.txt"  # Общий TXT файл

    results = []
    full_text_lines = []  # Сюда будем собирать строки для общего .txt

    with open(CSV_FILENAME, 'r', encoding="utf-8") as csvfile:
        # Считываем первую строку вручную и нормализуем заголовки
        first_line = csvfile.readline()
        headers = [h.strip().strip(";") for h in first_line.split(";") if h.strip()]
        reader = csv.DictReader(csvfile, fieldnames=headers, delimiter=";")

        for row in reader:
            drug_name = row.get("name", "").strip()
            link = row.get("link", "").strip().rstrip(";")
            if drug_name and link:
                text = get_text_from_rlsnet(link)
                if text:
                    # Сохраняем в JSON как список строк
                    results.append({drug_name: text.split("\n")})
                    # Добавляем в общий TXT файл
                    full_text_lines.append(text)
                else:
                    results.append({drug_name: None})

    # Сохраняем JSON
    with open(OUTPUT_FILENAME, 'w', encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)

    # Сохраняем общий TXT
    with open(TXT_OUTPUT_FILENAME, 'w', encoding="utf-8") as txtfile:
        txtfile.write("\n".join(full_text_lines))

    print(f"Результаты сохранены в {OUTPUT_FILENAME}")
    print(f"Общий текстовый файл сохранён в {TXT_OUTPUT_FILENAME}")