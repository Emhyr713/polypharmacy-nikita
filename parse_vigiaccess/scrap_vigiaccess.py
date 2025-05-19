import time
import unicodedata
import csv
import logging
from datetime import datetime
import os

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from concurrent.futures import ThreadPoolExecutor, as_completed

# Пути
GECKO_PATH = "parse_vigiaccess\\driver\\geckodriver.exe"
FIREFOX_BINARY_PATH = "C:\\Program Files\\Mozilla Firefox\\firefox.exe"

# Путь сохранений
DIR_SAVE = "parse_vigiaccess\\data\\vigiaccess_html"

WAIT_TIME = 10

RETRY = -1
DELAY_MULTIPLIER = 1.5

class ColorFormatter(logging.Formatter):
    """Форматтер с цветным выводом для консоли"""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[31;1m',
        'SUCCESS': '\033[35m',
        'RESET': '\033[0m'
    }

    def format(self, record):
        if record.levelname in self.COLORS:
            record.msg = f"{self.COLORS[record.levelname]}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logger():
    """Настройка комплексного логгера с выводом в файл и консоль"""
    logger = logging.getLogger('VigiAccessScraper')
    logger.setLevel(logging.DEBUG)
    
    # Создаем директорию для логов
    os.makedirs('parse_vigiaccess\\logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Файловый обработчик (все сообщения)
    file_handler = logging.FileHandler(
        filename=f'parse_vigiaccess\\logs\\scraping_{timestamp}.log', 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Консольный обработчик (только INFO и выше)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColorFormatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Добавляем обработчики
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Добавляем кастомный уровень SUCCESS
    logging.SUCCESS = 25  # Между INFO и WARNING
    logging.addLevelName(logging.SUCCESS, 'SUCCESS')
    
    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.SUCCESS):
            self._log(logging.SUCCESS, message, args, **kwargs)
    
    logging.Logger.success = success
    
    return logger

logger = setup_logger()
def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if c.isalpha()).lower()

def take_screenshot(driver, prefix="screenshot"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parse_vigiaccess\\logs\\scrennshots\\{prefix}_{timestamp}.png"
    driver.save_screenshot(filename)

def setup_driver():
    options = Options()
    options.add_argument("--headless")  # Режим без отображения окна браузера
    options.binary_location = FIREFOX_BINARY_PATH
    service = Service(GECKO_PATH)
    return webdriver.Firefox(service=service, options=options)

def accept_terms(driver):
    try:
        label = WebDriverWait(driver, WAIT_TIME).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "label[for='accept-terms-and-conditions']"))
        )
        label.click()
        driver.execute_script(
            "document.getElementById('accept-terms-and-conditions').dispatchEvent(new Event('change'));"
        )
        logger.success("Чекбокс условий отмечен")
    except Exception as e:
        logger.error(f"Ошибка при клике на чекбокс: {str(e)}")

def click_search_database(driver):
    try:
        search_button = WebDriverWait(driver, WAIT_TIME).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[dtid='caveat-acceptbutton']"))
        )
        search_button.click()
        logger.success("Кнопка 'Search database' нажата")
    except:
        driver.execute_script(
            "document.querySelector('button[dtid=\"caveat-acceptbutton\"]').disabled = false;"
        )
        driver.find_element(By.CSS_SELECTOR, "button[dtid='caveat-acceptbutton']").click()
        logger.success("Кнопка разблокирована и нажата")

def search_drug(driver, query):
    try:
        search_input = WebDriverWait(driver, WAIT_TIME).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[dtid='dashboard-drugsearch']"))
        )
        search_input.clear()
        search_input.send_keys(query)
        logger.info(f"Введен запрос: '{query}'")
        time.sleep(2*DELAY_MULTIPLIER)
    except Exception as e:
        logger.error(f"Ошибка при вводе запроса: {str(e)}")

def click_search_button(driver):
    try:
        search_btn = WebDriverWait(driver, WAIT_TIME).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a[dtid='dashboard-drugsearchbutton']"))
        )
        search_btn.click()
        logger.success("Кнопка 'Search' нажата")
        time.sleep(2*DELAY_MULTIPLIER)
    except Exception as e:
        logger.error(f"Ошибка при нажатии кнопки Search: {str(e)}")

def has_no_results(driver):
    try:
        # Ждём появления модального окна с "No results found"
        modal = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.modal-card"))
        )

        # Проверяем наличие секции с сообщением об отсутствии результатов
        no_results_section = modal.find_element(By.CSS_SELECTOR, "section[dtid='dashboard-noresults']")
        if "No results found" in no_results_section.text:
            logger.warning("Обнаружено сообщение 'No results found'")

            # Закрываем модалку через кнопку закрытия
            close_button = modal.find_element(By.CSS_SELECTOR, "button.delete")
            close_button.click()
            logger.info("Модальное окно закрыто")
            return True  # Результатов нет

    except Exception as e:
        # Модалка не появилась или другой элемент не найден
        pass

    return False  # Результаты есть

def handle_modal_ok(driver):
    try:
        modal_ok = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.modal.is-active button[dtid='common-ok']"))
        )
        modal_ok.click()
        logger.info("Нажата кнопка 'Ok' в модальном окне")
    except Exception as e:
        logger.debug(f"Модальное окно не появилось: {str(e)}")

def select_drug_from_list(driver, name, sublabel=None):
    find_drug_list = []
    try:
        WebDriverWait(driver, WAIT_TIME).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "tbody[dtid='dashboard-drugsearchresult'] tr"))
        )
        
        rows = driver.find_elements(By.CSS_SELECTOR, "tbody[dtid='dashboard-drugsearchresult'] tr")
        name_norm = normalize(name)
        sublabel_norm = normalize(sublabel) if sublabel else None

        for row in rows:
            divs = row.find_elements(By.CSS_SELECTOR, "td > div")
            if not divs:
                continue
                
            main_text = normalize(divs[0].text)
            sub_text = normalize(divs[1].text) if len(divs) > 1 else ""
            find_drug_list.append((main_text, sub_text))
            
            # Приоритетный поиск по sublabel если он задан
            if sublabel_norm and sublabel_norm in sub_text:
                divs[1].click()
                logger.info(f"✓ Найден по подписи: {main_text} / {sub_text}")
                handle_modal_ok(driver)
                return True
                
            # Поиск только по name если sublabel не задан или не найден
            if name_norm in main_text:
                divs[0].click()
                logger.info(f"✓ Найден по имени: {main_text}")
                handle_modal_ok(driver)
                return True

        # Если ничего не найдено
        logger.info(f"⚠️ Подходящий вариант не найден для: {name_norm}/{sublabel_norm}")
        logger.info(f"⚠️ Список препаратов в vigiaccess: {find_drug_list}")
        try:
            driver.find_element(By.CSS_SELECTOR, "div.modal-card button.delete").click()
        except:
            pass
        return False

    except Exception as e:
        logger.error(f"❌ Ошибка при выборе варианта: {e}")
        try:
            driver.find_element(By.CSS_SELECTOR, "div.modal-card button.delete").click()
        except:
            pass
        return False

# Функция для нажатия на вкладку "Table" для блока "Age group distribution" и "Patient sex distribution"
def click_table_tab(driver, block_name, block_id):
    """Клик по вкладке Table для указанного блока"""
    try:
        tab = WebDriverWait(driver, WAIT_TIME).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f"div[dtid='{block_id}'] li[dtid='dashboard-tabletab'] a"))
        )
        tab.click()
        logger.success(f"Вкладка 'Table' для блока '{block_name}' нажата")
    except Exception as e:
        logger.error(f"Ошибка при нажатии вкладки '{block_name}': {str(e)}")

def click_all_load_more(driver):
    while True:
        try:
            # Пытаемся найти все кнопки с ожиданием
            load_more_buttons = WebDriverWait(driver, WAIT_TIME).until(
                EC.presence_of_all_elements_located((By.XPATH, '//*[starts-with(@dtid, "dashboard-socrowloadmore")]'))
            )
            
            if not load_more_buttons:
                logger.debug("Кнопки 'Load More' не найдены")
                return True

            # Кликаем по всем найденным кнопкам
            for button in load_more_buttons:
                try:
                    button.click()
                    if driver.find_elements(By.CSS_SELECTOR, 'div.toast.toast-error'):
                        logger.error("Обнаружено сообщение об ошибке")
                        return RETRY
                    time.sleep(0.5*DELAY_MULTIPLIER)  # Пауза между кликами
                except Exception as e:
                    logger.warning(f"Ошибка при клике на кнопку: {str(e)}")
                    return RETRY

            # Дополнительное ожидание после кликов
            time.sleep(2*DELAY_MULTIPLIER)

        except TimeoutException:
            logger.debug(f"Кнопки 'Load More' не появились за {WAIT_TIME} сек - вероятно их нет")
            return True
        except Exception as e:
            logger.error(f"Неожиданная ошибка в click_all_load_more: {str(e)}")
            return RETRY

def click_all_icons(driver):
    try:
        if not (icons := WebDriverWait(driver, WAIT_TIME).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'span.icon i.fas.fa-angle-right'))
            )):
            logger.info("Нет иконок для раскрытия")
            return 0
            
        logger.info(f"Найдено {len(icons)} иконок")
        
        for icon in icons:
            try:
                # Прокручиваем к элементу перед кликом
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", icon)
                
                # Ждём, пока элемент станет кликабельным
                WebDriverWait(driver, WAIT_TIME).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'span.icon i.fas.fa-angle-right'))
                )
                
                # Альтернативный способ клика через JavaScript
                driver.execute_script("arguments[0].click();", icon)
                time.sleep(0.5*DELAY_MULTIPLIER)
                
                if driver.find_elements(By.CSS_SELECTOR, 'div.toast.toast-error'):
                    logger.error("Обнаружено сообщение об ошибке")
                    return RETRY

            except Exception as e:
                logger.error(f"Ошибка клика: {e}")
                return RETRY
        
        logger.success("Все иконки успешно обработаны")
        return len(icons)
            
    except Exception as e:
        logger.error(f"Ошибка при поиске иконок: {e}")
        return RETRY

def save_html(driver, query, dir=""):
    try:
        # Корректная обработка пути для сохранения файла
        if dir:
            os.makedirs(dir, exist_ok=True)
            query = query.replace(";", "+")
            file_path = os.path.join(dir, f"all_html_{query}.html")
        else:
            file_path = f"all_html_{query}.html"

        # Список блоков, которые мы будем искать и сохранять
        blocks = []

        notification_box = None
        try:
            notifications = driver.find_elements(By.CSS_SELECTOR, "div.notification.has-background-info")
            for box in notifications:
                text = box.text.strip()
                if not text.startswith("Note:"):  # отсекаем "неправильный" блок
                    notification_box = box
                    blocks.append(notification_box.get_attribute("outerHTML"))
                    logger.debug(f"Блок 'Notification' сохранен")
                    break
            else:
                logger.warning(f"Блок 'Notification' не найден")

        except Exception as e:
            logger.warning(f"❌ Ошибка при поиске блока Notification: {e}")

        # Блок с Reported potential side effects
        target_box = None
        try:
            all_boxes = driver.find_elements(By.CSS_SELECTOR, "div.box")
            for box in all_boxes:
                try:
                    title = box.find_element(By.CSS_SELECTOR, "h1.title").text.strip()
                    if title == "Reported potential side effects":
                        target_box = box
                        blocks.append(target_box.get_attribute("outerHTML"))
                        logger.debug(f"Блок 'Reported potential side effects' сохранен")
                        break
                except:
                    continue
        except Exception as e:
            logger.warning(f"Блок 'Reported potential side effects' не найден: {str(e)}")
            

        # Блок с Patient sex distribution
        try:
            sex_distribution_box = driver.find_element(By.CSS_SELECTOR, "div.box[dtid='dashboard-sexdist']")
            blocks.append(sex_distribution_box.get_attribute("outerHTML"))
            logger.debug(f"Блок 'Patient sex distribution' сохранен")
        except Exception as e:
            logger.warning(f"Блок 'Patient sex distribution' не найден: {str(e)}")

        # Блок с Age group distribution
        try:
            age_group_distribution_box = driver.find_element(By.CSS_SELECTOR, "div.box[dtid='dashboard-agegroupdist']")
            blocks.append(age_group_distribution_box.get_attribute("outerHTML"))
            logger.debug(f"Блок 'Age group distribution' сохранен")
        except Exception as e:
            logger.warning(f"Блок 'Age group distribution' не найден: {str(e)}")

        # Если хотя бы один блок найден, сохраняем их
        if blocks:
            html_content = '\n'.join(blocks)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(html_content)
            logger.success(f"✓ Все блоки сохранены в файл {file_path} для {query}")
            return True
        else:
            logger.warning(f"Не найдено ни одного блока для {query}")
            return False

    except Exception as e:
        logger.error(f"Ошибка при сохранении HTML: {str(e)}")
        return False

def check_terms_and_conditions(driver):
    try:
        # Проверяем, существует ли чекбокс "accept-terms-and-conditions" на странице
        checkbox = driver.find_element(By.CSS_SELECTOR, "input#accept-terms-and-conditions")
        return True
    except Exception as e:
        return False

def process_drug(driver, drug, sublabel=None, attempt=1):
    max_attempts = 5
    if attempt > max_attempts:
        logger.error(f"Достигнуто максимальное количество попыток ({max_attempts}) для препарата {drug}")
        return False

    logger.info(f"\nПопытка {attempt}/{max_attempts}: начинаем обработку препарата: {drug}")

    if check_terms_and_conditions(driver):
        accept_terms(driver)
        click_search_database(driver)

    search_drug(driver, drug)
    click_search_button(driver)

    if has_no_results(driver):
        logger.warning(f"Препарат {drug} не найден")
        return False

    if not select_drug_from_list(driver, name=drug, sublabel=sublabel):
        logger.warning(f"Не удалось выбрать препарат {drug} из списка")
        return False

    time.sleep(2*DELAY_MULTIPLIER)
    click_table_tab(driver, "Age group distribution", "dashboard-agegroupdist")
    click_table_tab(driver, "Patient sex distribution", "dashboard-sexdist")

    result = click_all_icons(driver)
    # logger.info(f"click_all_icons выполнилось с {result}")
    if result == RETRY:
        logger.warning(f"Ошибка при клике иконок, пробуем снова")
        time.sleep(5*DELAY_MULTIPLIER)
        take_screenshot(driver)
        driver.refresh()
        return process_drug(driver, drug, sublabel, attempt+1)

    result = click_all_load_more(driver)
    # logger.info(f"click_all_icons выполнилось с {result}")
    if result == RETRY:
        logger.warning(f"Ошибка при клике 'Load More', пробуем снова")
        time.sleep(5*DELAY_MULTIPLIER)
        take_screenshot(driver)
        driver.refresh()
        return process_drug(driver, drug, sublabel, attempt+1)

    if not save_html(driver, drug, dir=DIR_SAVE):
        return False
        
    logger.success(f"Успешно завершена обработка {drug}")
    return True

def process_single_drug(drug_row):
    """Обработка одного препарата в отдельном потоке"""
    driver = None
    try:
        driver = setup_driver()
        driver.get("https://vigiaccess.org/")
        
        drug_name_en = drug_row.get('drug_name_en', '').strip()
        drug_name_ru = drug_row.get('drug_name_ru', '').strip()
        drug_subname_en = None

        if "+" in drug_name_en:
            drug_name_en = drug_name_en.replace("+", ";")
            drug_subname_en = drug_name_en

        if "Ferric carboxymaltose" == drug_name_en:
            drug_name_en = "Iron"
            drug_subname_en = "Iron"

        success = process_drug(driver, drug_name_en, drug_subname_en)
        
        return (drug_name_en, drug_name_ru, success)
        
    except Exception as e:
        logger.critical(f"Фатальная ошибка в потоке: {str(e)}")
        return (drug_row.get('drug_name_en', ''), drug_row.get('drug_name_ru', ''), False)
    finally:
        if driver:
            driver.quit()


def main():
    LINKS_SIDE_EFFECT_FILENAME = "make_side_effect_dataset\\data\\drugs_table4.csv"
    not_found_log = "parse_vigiaccess\\data\\not_found_drugs.txt"

    logger.info("Запуск скрипта")

    with open(LINKS_SIDE_EFFECT_FILENAME, newline='', encoding='utf-8') as csvfile:
        drug_rows = list(csv.DictReader(csvfile, delimiter=';'))
        logger.info(f"Загружено {len(drug_rows)} препаратов для обработки")

    with ThreadPoolExecutor(max_workers=4) as executor, \
            open(not_found_log, "w", encoding="utf-8") as log_file:
            
            futures = [executor.submit(process_single_drug, row) for row in drug_rows]
            logger.info(f"Запущено {len(futures)} задач")

            for future in as_completed(futures):
                drug_name_en, drug_name_ru, success = future.result()
                if not success:
                    log_file.write(f"{drug_name_en} ({drug_name_ru})\n")
                    log_file.flush()
                    logger.warning(f"Пропущен препарат: {drug_name_en}")


if __name__ == "__main__":
    main()
