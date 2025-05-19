from bs4 import BeautifulSoup
import re
import time
from collections import defaultdict
from pprint import pprint

class Drug:
    def __init__(self, name_ru=None, name_en=None):
        self.name_ru = name_ru
        self.name_en = name_en
        
        self.group_side_e = None
        self.age_info = None
        self.sex_info = None
        self.report_count = None


    def _parse_side_effects(self, soup):
        group_data = defaultdict(lambda: {"side_e_dict": {}, "precent": None, "cases": None})
        current_group = None

        for li in soup.find_all('li'):
            text = li.get_text(strip=True)
            spans = li.find_all('span')

            if len(spans) > 1:
                group, percent, cases = self.parse_main_info(text)
                current_group = group
                group_data[current_group]["precent"] = percent
                group_data[current_group]["cases"] = cases
            elif current_group:
                side_effect, cases = self.add_side_e(text)
                group_data[current_group]["side_e_dict"][side_effect] = cases

        return dict(group_data)

    def _parse_report_count(self, soup):
        # Преобразуем в текст, удаляя невидимые юникод-символы (но не весь non-ASCII)
        text = str(soup).replace('\u202f', '')\
                        .replace('\u200b', '')\
                        .replace('\u200c', '')\
                        .replace('\u200e', '')\
                        .replace('\u2060', '')
        
        # Ищем число в шаблоне: "There are <strong>55 324</strong> reports"
        match = re.search(r"There (?:is|are)\s*<strong>([\d\s]+)</strong>\s*reports?", text)
        if match:
            number_str = match.group(1).replace(" ", "")
            try:
                return int(number_str)
            except ValueError:
                return None
        return None

    def _parse_stat_table(self, soup):

        stat_list = {}

        for tr in soup.find_all('tr'):
            cells = tr.find_all('td')
            if len(cells) >= 3:
                label = cells[0].get_text(strip=True)
                count_text = re.sub(r'[^\x00-\x7F]+', '', cells[1].get_text())
                percentage = cells[2].get_text(strip=True)
                try:
                    count = int(count_text)
                except ValueError:
                    print(f"Ошибка преобразования количества: {count_text}")
                    count = 0
                stat_list[label] = count
        return stat_list

    def parse_vigiaccess_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')

        # Reported potential side effects
        side_effects_block = soup.find("h1", string="Reported potential side effects")

        if side_effects_block:
            self.group_side_e = self._parse_side_effects(side_effects_block.find_parent("div", class_="box"))

        # Notification info (по классу)
        notification_block = soup.find("div", class_="notification has-background-info")
        if notification_block:
            self.report_count = self._parse_report_count(soup)

        # Поиск блока с полом
        sex_block = soup.find('div', {'class': 'box', 'dtid': 'dashboard-sexdist'})
        if sex_block:
            self.sex_info = self._parse_stat_table(sex_block)

        # Поиск блока с возрастом
        age_block = soup.find('div', {'class': 'box', 'dtid': 'dashboard-agegroupdist'})
        if age_block:
            self.age_info = self._parse_stat_table(age_block)

    def parse_main_info(self, text):
        visible_text = re.sub(r'[^\x00-\x7F]+', '', text)
        match = re.search(r'(.+)\s+\((\d+)%,\s*(\d+)\s+ADRs\)', visible_text)
        if match:
            group_side_e = match.group(1).strip()
            percent = int(match.group(2))
            cases = int(match.group(3))
            return group_side_e, percent, cases
        else:
            raise ValueError(f"Cannot parse side effect main info: {text}")

    def add_side_e(self, text):
        visible_text = re.sub(r'[^\x00-\x7F]+', '', text)
        match = re.search(r'(.+)\s*\((\d+)\)\s*', visible_text)
        if match:
            side_e = match.group(1).strip()
            cases = int(match.group(2))
            return side_e, cases
        else:
            print(f"Warning: Cannot parse sub side effect: {text}")


class GroupSideEffect:
    def __init__(self, text: str):
        self.name, self.percents, self.cases = self.parse_main_info(text)
        self.side_effects = []



    def __str__(self):
        base = f"* {self.name} ({self.percents}%, {self.cases} ADRs)"
        subs = "\n".join(f"  - {sub}" for sub in self.side_effects)
        return f"{base}\n{subs}" if self.side_effects else base

class SideEffect:
    def __init__(self, name: str, cases: int):
        self.name = name
        self.cases = cases

    def __str__(self):
        return f"{self.name} ({self.cases})"
    
class DrugStatRow:
    def __init__(self, label: str, count: int, percentage: str):
        self.label = label
        self.count = count
        self.percentage = percentage

    def __str__(self):
        return f"{self.label} / {self.count} / {self.percentage}"

if __name__ == "__main__":

    DIR_SOURCE_HTML  = "parse_vigiaccess\\data\\vigiaccess_html"

    drug_name_en = "amiodarone"

    with open(f"{DIR_SOURCE_HTML}\\all_html_{drug_name_en}.html", 'r', encoding='utf-8') as file:
        html_content = file.read()

    drug_obj = Drug(name_en=drug_name_en)
    drug_obj.parse_vigiaccess_html(html_content)




