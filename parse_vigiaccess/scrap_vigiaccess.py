from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import time
import unicodedata
import csv

import os

# Пути
GECKO_PATH = "parse_vigiaccess\\driver\\geckodriver.exe"
FIREFOX_BINARY_PATH = "C:\\Program Files\\Mozilla Firefox\\firefox.exe"

# Путь сохранений
DIR_SAVE = "parse_vigiaccess\\data\\vigiaccess_html"

WAIT_TIME = 5

def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if c.isalpha()).lower()

def setup_driver():
    options = Options()
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
        print("✓ Чекбокс отмечен")
    except Exception as e:
        print("❌ Ошибка при клике на чекбокс:", e)

def click_search_database(driver):
    try:
        search_button = WebDriverWait(driver, WAIT_TIME).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[dtid='caveat-acceptbutton']"))
        )
        search_button.click()
        print("✓ Кнопка 'Search database' нажата!")
    except:
        driver.execute_script(
            "document.querySelector('button[dtid=\"caveat-acceptbutton\"]').disabled = false;"
        )
        driver.find_element(By.CSS_SELECTOR, "button[dtid='caveat-acceptbutton']").click()
        print("✓ Кнопка разблокирована через JS и нажата")

def search_drug(driver, query):
    try:
        search_input = WebDriverWait(driver, WAIT_TIME).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[dtid='dashboard-drugsearch']"))
        )
        search_input.clear()
        search_input.send_keys(query)
        print(f"✓ Введено '{query}'")
        time.sleep(2)
    except Exception as e:
        print("❌ Ошибка при вводе лекарства:", e)

def click_search_button(driver):
    try:
        search_btn = WebDriverWait(driver, WAIT_TIME).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a[dtid='dashboard-drugsearchbutton']"))
        )
        search_btn.click()
        print("✓ Кнопка 'Search' нажата")
        time.sleep(2)
    except Exception as e:
        print("❌ Ошибка при нажатии кнопки Search:", e)

def has_no_results(driver):
    try:
        # Ждём появления модального окна с "No results found"
        modal = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.modal-card"))
        )

        # Проверяем наличие секции с сообщением об отсутствии результатов
        no_results_section = modal.find_element(By.CSS_SELECTOR, "section[dtid='dashboard-noresults']")
        if "No results found" in no_results_section.text:
            print("⚠️ Обнаружено сообщение 'No results found'")

            # Закрываем модалку через кнопку закрытия
            close_button = modal.find_element(By.CSS_SELECTOR, "button.delete")
            close_button.click()
            print("✓ Модальное окно закрыто")
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
        print("✓ Нажата кнопка 'Ok' в модальном окне")
    except Exception as e:
        print("ℹ️ Модальное окно не появилось или не обнаружено:", e)

def select_drug_from_list(driver, name, sublabel=None):
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
            sub_text = normalize(divs[1].text) if len(divs) > 1 else None

            if name_norm in main_text:
                if sublabel_norm:
                    if sub_text and sublabel_norm in sub_text:
                        divs[0].click()
                        print(f"✓ Найден и выбран вариант с подписью: {main_text} / {sub_text}")
                        handle_modal_ok(driver)
                        return True
                else:
                    divs[0].click()
                    print(f"✓ Найден и выбран вариант: {main_text}")
                    handle_modal_ok(driver)
                    return True

        # Вариант не найден — пробуем закрыть модалку
        print("⚠️ Подходящий вариант не найден, пробуем закрыть модалку")
        try:
            modal = driver.find_element(By.CSS_SELECTOR, "div.modal-card")
            close_button = modal.find_element(By.CSS_SELECTOR, "button.delete")
            close_button.click()
            print("✓ Модальное окно закрыто")
        except Exception as close_err:
            print("❌ Ошибка при закрытии модального окна:", close_err)

        return False

    except Exception as e:
        print("❌ Ошибка при выборе варианта:", e)
        try:
            modal = driver.find_element(By.CSS_SELECTOR, "div.modal-card")
            close_button = modal.find_element(By.CSS_SELECTOR, "button.delete")
            close_button.click()
            print("✓ Модальное окно закрыто")
        except Exception as close_err:
            print("❌ Ошибка при закрытии модального окна:", close_err)
        return False

# Функция для нажатия на вкладку "Table" для блока "Age group distribution"
def click_age_group_table_tab(driver):
    try:
        age_group_table_tab = WebDriverWait(driver, WAIT_TIME).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div[dtid='dashboard-agegroupdist'] li[dtid='dashboard-tabletab'] a"))
        )
        age_group_table_tab.click()
        print("✓ Нажата вкладка 'Table' в блоке 'Age group distribution'")
    except Exception as e:
        print("❌ Ошибка при нажатии на вкладку 'Table' в блоке 'Age group distribution':", e)

# Функция для нажатия на вкладку "Table" для блока "Patient sex distribution"
def click_patient_sex_table_tab(driver):
    try:
        patient_sex_table_tab = WebDriverWait(driver, WAIT_TIME).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div[dtid='dashboard-sexdist'] li[dtid='dashboard-tabletab'] a"))
        )
        patient_sex_table_tab.click()
        print("✓ Нажата вкладка 'Table' в блоке 'Patient sex distribution'")
    except Exception as e:
        print("❌ Ошибка при нажатии на вкладку 'Table' в блоке 'Patient sex distribution':", e)

def click_load_more(driver):
    try:
        load_more_button = WebDriverWait(driver, WAIT_TIME).until(
            EC.element_to_be_clickable((By.XPATH, '//*[starts-with(@dtid, "dashboard-socrowloadmore")]'))
        )
        load_more_button.click()
        print('Клик по "Load more"')
        return True
    except:
        print('Кнопка "Load more" больше не найдена.')
        return False

def click_all_icons(driver):
    try:
        icons = driver.find_elements(By.CSS_SELECTOR, 'span.icon i.fas.fa-angle-right')
        print(f'Найдено {len(icons)} иконок для раскрытия')
        for icon in icons:
            icon.click()
            # time.sleep(0.3)
        print('Все иконки раскрыты.')
    except Exception as e:
        print(f"Ошибка при раскрытии иконок: {e}")

def click_all_load_more(driver):
    while True:
        if not click_load_more(driver):
            break
        time.sleep(0.5)


def save_html(driver, query, dir=""):
    try:
        # Корректная обработка пути для сохранения файла
        if dir:
            os.makedirs(dir, exist_ok=True)
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
                    print("✓ Блок 'Notification' добавлен в HTML")
                    break
            else:
                print("⚠️ Подходящий блок 'Notification' не найден")

        except Exception as e:
            print(f"❌ Ошибка при поиске блока Notification: {e}")

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
                        print("✓ Блок 'Reported potential side effects' добавлен в HTML")
                        break
                except:
                    continue
        except:
            print("⚠️ Не удалось найти блок 'Reported potential side effects'")

        # Блок с Patient sex distribution
        try:
            sex_distribution_box = driver.find_element(By.CSS_SELECTOR, "div.box[dtid='dashboard-sexdist']")
            blocks.append(sex_distribution_box.get_attribute("outerHTML"))
            print("✓ Блок 'Patient sex distribution' добавлен в HTML")
        except:
            print("⚠️ Блок 'Patient sex distribution' не найден")

        # Блок с Age group distribution
        try:
            age_group_distribution_box = driver.find_element(By.CSS_SELECTOR, "div.box[dtid='dashboard-agegroupdist']")
            blocks.append(age_group_distribution_box.get_attribute("outerHTML"))
            print("✓ Блок 'Age group distribution' добавлен в HTML")
        except:
            print("⚠️ Блок 'Age group distribution' не найден")

        # Если хотя бы один блок найден, сохраняем их
        if blocks:
            html_content = '\n'.join(blocks)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(html_content)
            print(f"✓ Все блоки сохранены в файл {file_path} для {query}")
        else:
            print(f"⚠️ Не удалось найти ни один блок для {query}")

    except Exception as e:
        print(f"❌ Ошибка при сохранении HTML для {query}: {e}")

def check_terms_and_conditions(driver):
    try:
        # Проверяем, существует ли чекбокс "accept-terms-and-conditions" на странице
        checkbox = driver.find_element(By.CSS_SELECTOR, "input#accept-terms-and-conditions")
        return True
    except:
        return False

def process_drug(driver, drug, sublabel=None):
    print(f"\nНачинаем обработку препарата: {drug}")
    
    if check_terms_and_conditions(driver):
        accept_terms(driver)
        click_search_database(driver)

    search_drug(driver, drug)
    click_search_button(driver)

    if has_no_results(driver):
        return False

    if not select_drug_from_list(driver, name=drug, sublabel=sublabel):
        return False

    time.sleep(2)
    click_age_group_table_tab(driver)
    click_patient_sex_table_tab(driver)
    click_all_icons(driver)
    click_all_load_more(driver)

    save_html(driver, drug, dir=DIR_SAVE)
    # time.sleep(2)
    return True

def main():
    driver = setup_driver()
    driver.get("https://vigiaccess.org/")

    LINKS_SIDE_EFFECT_FILENAME = "make_side_effect_dataset\\data\\drugs_table4.csv"
    not_found_log = "parse_vigiaccess\\data\\not_found_drugs.txt"

    # Читаем CSV-файл
    with open(LINKS_SIDE_EFFECT_FILENAME, newline='', encoding='utf-8') as csvfile:
        drug_rows = list(csv.DictReader(csvfile, delimiter=';'))  # Читаем сразу весь файл

    with open(not_found_log, "w", encoding="utf-8") as log_file:
        for row in drug_rows:
            drug_name_ru = row.get('drug_name_ru', '').strip()
            drug_name_en = row.get('drug_name_en', '').strip()
            drug_subname_en = None

            if "+" in drug_name_en:
                drug_name_en = drug_name_en.replace("+", ";")
                drug_subname_en = drug_name_en

            if "Ferric carboxymaltose" == drug_name_en:
                drug_subname_en = drug_name_en

            success = process_drug(driver, drug_name_en, drug_subname_en)
            if not success:
                log_file.write(f"{drug_name_en} ({drug_name_ru})\n")
                log_file.flush()  # На случай прерывания

    driver.quit()

if __name__ == "__main__":
    main()
