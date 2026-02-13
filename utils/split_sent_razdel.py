from razdel import sentenize
import re

def split_format_text(text, delete_parentheses_flag=True, filter_line_flag=True):

    if delete_parentheses_flag:
        text = re.sub(r'\(.*?\)', '', text)                             # Удаление круглых скобок и их содержимого

    text = re.sub(r'^Таблица.*\d+$', '', text, flags=re.MULTILINE)      # "Таблица n"
    text = re.sub(r'^RxList\.com.*$', '', text, flags=re.MULTILINE)     # "RxList.com ..."
    text = re.sub(r',? показаны в таблице \d+', '', text)               # ", показаны в таблице n"
    text = re.sub(r'представлен(?:а|о|ы)? в таблице \d+', '', text)     # "представлен(а)(о)(ы) в таблице n"
    text = re.sub(r'приведен(?:а|о|ы)? в таблице \d+', '', text)        # "приведен(а)(о)(ы) в таблице n"
    text = re.sub(r'\((см\.\s*раздел.*?)\)', '', text)                  # "(см.раздел ...)"

    text = re.sub(r'Результаты представлены в таблице \d+.', '', text)  # "Результаты представлены в таблице n"
    
    # Удаление лишних пробелов
    text = re.sub(r' +\,', ',', text)
    text = re.sub(r' +\.', '.', text)
    text = re.sub(r' +\:', ':', text)
    text = re.sub(r' +\;', ';', text)
    text = re.sub(r'\,\.', '.', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace('в т. ч.', 'в т.ч.')

    # Разделение на предложения
    sentenses = [sentense.text for sentense in list(sentenize(text))]
    # Проставление \n
    text = '\n'.join(sentenses)

    text = re.sub(r'\n\n+', '\n', text)

    text = text.replace('\n,', ',')

    # Доп разделение  
    text = text.replace(' ч. ', ' ч.\n')
    text = text.replace('/ч. ', '/ч.\n')
    text = text.replace(' тел. ', ' тел.\n')
    text = text.replace(' с. ', ' с.\n')
    
    def filter_line(text):

        # Фильтрация строк, заканчивающихся не на ".", ";" или ":"
        lines = text.split('\n')

        # Функция для проверки, содержит ли строка только одно слово с точкой
        def is_single_word_with_period(line):
            words = line.split()
            return len(words) == 1 and words[0].endswith('.')

        # Фильтрация строк, которые заканчиваются на ".", ";", ":"
        # и не являются одним словом с точкой
        filtered_lines = [
            line for line in lines
            if (line.endswith(('.', ';', ':'))
                and
                not is_single_word_with_period(line))
        ]

        text = '\n'.join(filtered_lines)

        return text

    # Фильтрация строк
    if filter_line_flag:
        text = filter_line(text)

    return text