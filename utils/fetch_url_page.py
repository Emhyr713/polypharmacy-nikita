import requests
from bs4 import BeautifulSoup
import time
import random

def fetch_page_with_retries(url, max_retries=5, headers=None):
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive"
        }

    retries = 0

    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Проверка статуса
            return BeautifulSoup(response.content, 'html.parser')

        except requests.exceptions.RequestException as e:
            wait_time = 2 ** retries + random.uniform(0, 1)  # экспоненциальная задержка
            print(f"Попытка {retries + 1}: ошибка {e}. Повтор через {round(wait_time, 2)} сек...")
            time.sleep(wait_time)
            retries += 1

    print(f"Не удалось загрузить {url} после {max_retries} попыток.")
    return None