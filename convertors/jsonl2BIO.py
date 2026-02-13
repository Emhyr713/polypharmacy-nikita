import re
import json

from razdel import sentenize

# import spacy
# from spacy.tokenizer import Tokenizer
# from spacy.tokens import Span
# from spacy.tokens import Doc

from itertools import combinations

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MyToken:
    text: str
    start: int
    end: int

@dataclass
class MySpan:
    id: Optional[int] = None
    doc: Optional["MyDoc"] = None   # Устанавливается после инициализации doc
    start: int = 0                  # Начало в символах
    end: int = 0                    # Конец в символах
    label: str = ""                 # Метка
    tokens: List[MyToken] = field(default_factory=list)  # Список токенов
    _text: str = ""  # Текст Span, который будет извлечён позже из токенов

    def __post_init__(self):
        if self.doc:
            # Инициализируем tokens после того, как doc уже проинициализирован
            self.tokens = self.tokens_in_range
            # Сохраняем текст как срез из doc.text
            self._text = self.doc.text[self.start:self.end]

    @property
    def tokens_in_range(self) -> List[MyToken]:
        if not self.doc:
            raise ValueError("Span is not associated with any document.")
        
        # Получаем токены, которые пересекаются с диапазоном [start, end)
        tokens_in_range = [
            token for token in self.doc.tokens
            if token.start >= self.start and token.end <= self.end
        ]
        
        # Обновляем start и end для корректности
        if tokens_in_range:
            self.start = tokens_in_range[0].start       # Начало первого токена
            self.end = tokens_in_range[-1].end     # Конец последнего токена
        return tokens_in_range

    @property
    def text(self) -> str:
        # Возвращаем текст как срез из doc.text, вычисленный по диапазону start и end
        return self._text

@dataclass
class MyDoc:
    text: str
    tokens: List[MyToken] = field(init=False)
    ents: List[MySpan] = field(default_factory=list)
    offset: int = 0

    def __post_init__(self):
        # Создаём токены из текста
        self.tokens = [
            MyToken(match.group(), match.start()+self.offset, match.end()+self.offset)
            for match in re.finditer(r'\w+|[^\w\s]', self.text)
        ]

    def add_span(self, start: int, end: int, label: str = "", id: int = None) -> MySpan:
        # Создаём Span и добавляем его в документ
        span = MySpan(doc=self, id = id, start=start, end=end, label=label)
        self.ents.append(span)
        return span


def write_add(file, text):
    with open(file, 'a', encoding='utf-8') as file:
        file.write(text)

def check_correct_token(doc, text, ent):
    """Проверка на целостность токена и корректировка границ токена с учётом символов пунктуации и открывающих/закрывающих символов."""

    # Разделение переданных данных
    id, start_const, end_const, label = ent
    start, end = start_const, end_const  # Используем start_const и end_const для изначальных границ

    if len(doc.text) < end:
        return None

    # Символы, которые нужно проверить (открывающие и закрывающие)
    open_symbols = '("«'  # Открывающие символы
    close_symbols = ')"»'  # Закрывающие символы
    punct_symbols = ',.;:-'  # Пунктуация для удаления

    # Подсчёт количества открывающих и закрывающих символов в пределах токена
    open_count = sum(text[start:end].count(symbol) for symbol in open_symbols)
    close_count = sum(text[start:end].count(symbol) for symbol in close_symbols)

    # Печать отладочной информации (можно убрать позже)
    print(f"text_ent: '{text[start:end]}', text_full:{len(doc.text)}, start-end:{start}-{end} open_count: {open_count}, close_count: {close_count}")
    
    # Определяем количество символов, которые нужно удалить с каждой стороны
    count_del_start = open_count - close_count
    count_del_end = close_count - open_count

    # Удаляем лишние пробелы и пунктуацию с конца и с начала
    while end > start and (text[end-1].isspace() or text[end-1] in punct_symbols):
        end -= 1
    while end > start and (text[start].isspace() or text[start] in punct_symbols):
        start += 1

    # Убрать лишние открывающие/закрывающие символы с учётом баланса
    if end > start and text[start] in open_symbols and count_del_start > 0:
        start += 1
    if end > start and text[end-1] in close_symbols and count_del_end > 0:
        end -= 1

    # Откатываем start и end, если перед ним есть буквы (нужно для целостности слова)
    while start > 0 and text[start-1].isalpha():
        start -= 1
    while end < len(text) and text[end].isalpha():
        end += 1

    # Создание Span с обновлёнными границами
    span = doc.add_span(start, end, label=label, id=id)

    # Если Span не был создан, выводим информацию о "браке"
    if not span:
        print(f"Брак: '{text[start_const:end_const]}' -> '{text[start:end]}'")

    return span


def span_filter(spans):
    """Удаление пересекающихся сущностей."""
    # Сортируем спаны: по индексу начала,
    sorted_spans = sorted(spans,
                          key=lambda span: (span.start))
    # Результирующий список без пересечений
    filtered_spans = []
    for span in sorted_spans:
        # Проверяем пересечения с последним добавленным в список спаном
        if filtered_spans and span.start < filtered_spans[-1].end:
            if len(filtered_spans[-1].text) < len(span.text):
                filtered_spans[-1] = span
        else:
            filtered_spans.append(span)
    return filtered_spans

# def generate_bio_tags(doc, sent_start, sent_end):
#     """Генерирует BIO-теги для токенов в пределах предложения."""
#     return [
#         (token.text, f'{token.ent_iob_}-{token.ent_type_}'
#          if token.ent_iob_ != 'O' else 'O')
#         for token in doc
#         if sent_start <= token.idx < sent_end
#     ]

def generate_bio_tags(doc, sent_start, sent_end):
    """Генерирует BIO-теги для токенов в пределах предложения."""

    # for ent in doc.ents: 
    #     print(ent.text)

    bio_tags = []

    token_tag_dict = {"tokens":[], "tags":[]}
    # tokens = []
    # tags = []

    # Пройдем по всем токенам в документе
    for token in doc.tokens:

        # Найдем спаны, которые пересекаются с данным токеном
        token_bio = "O"  # По умолчанию токен помечается как 'Outside'
        for span in doc.ents:
            # Если токен попадает в диапазон спана
            if token.start >= span.start and token.end <= span.end:
                # Если это первый токен в спане, то это 'B' (beginning)
                if token.start == span.start:
                    token_bio = f"B-{span.label}"
                # Если это токен внутри спана, то это 'I' (inside)
                elif token.start > span.start and token.end <= span.end:
                    token_bio = f"I-{span.label}"
                break  # Если нашли хотя бы один спан, больше не проверяем

        bio_tags.append((token.text, token_bio))

        token_tag_dict["tokens"].append(token.text)
        token_tag_dict["tags"].append(token_bio)
        # tokens.append(token.text)
        # tags.append(token.text)

    return bio_tags, token_tag_dict


def find_sentence_for_span(span, sentences):
    """Ищет предложение, в котором находится span."""
    for sentence in sentences:
        if  sentence.start <= span.start < sentence.stop:
            return sentence
    return None  # Если не найдено

def create_sentence_lookup(sentences):
    """Создаёт словарь для быстрого поиска предложения по диапазону символов."""
    sentence_lookup = {}
    for sentence in sentences:
        sentence_lookup[(sentence.start, sentence.end)] = sentence
    return sentence_lookup

def find_missing_relations(id_ent_list,
                           all_relations,
                           sentence_for_entity,
                           id2ent):
    """Ищет связи и связи, которых нет."""

    # Существующие связи
    existing_pairs = set(
        (item["from_id"], item["to_id"])
        for item in all_relations
        if item["from_id"] in id_ent_list
    )

    # Все возможные комбинации id сущностей
    all_combinations = combinations(id_ent_list, 2)

    # Поиск связей, относящихся к сущностям текущего предложения
    relations = [
        (
            id2ent[item["from_id"]].text,
            id2ent[item["from_id"]].label,
            sentence_for_entity.get(item["from_id"]),
            item["type"],
            id2ent[item["to_id"]].text,
            id2ent[item["to_id"]].label,
            sentence_for_entity.get(item["to_id"])
        )
        for item in all_relations
        if item["from_id"] in id_ent_list
        and item["to_id"] in id_ent_list
    ]

    # Находим комбинации, которых нет в sent_relations
    sent_missing_relations = [
        (
            id2ent[from_id].text,
            id2ent[from_id].label,
            sentence_for_entity.get(from_id),
            "Not_link",
            id2ent[to_id].text,
            id2ent[to_id].label,
            sentence_for_entity.get(to_id)
        )
        for from_id, to_id in all_combinations
        if (from_id, to_id) not in existing_pairs and (to_id, from_id) not in existing_pairs
        # and (id2ent[from_id].label != "side_e" and id2ent[to_id].label != "side_e")
        # and (id2ent[from_id].label != "side_e" and id2ent[to_id].label != "prepare")
        # and (id2ent[from_id].label != "side_e" and id2ent[to_id].label != "illness")
        # and (id2ent[from_id].label != "group" and id2ent[to_id].label != "group")
        # and id2ent[from_id].label != "illness"
        # and id2ent[from_id].label != "condition"
    ]
    
    return relations, sent_missing_relations

def link_ent_sent(data):
    """Разделение текста на предложения,"""
    """Назначение сущностей каждому предложению"""
    """Назначение связей    каждому предложению"""

    text        = data['text']
    entities    = data['entities']
    relations   = data['relations']
    
    info_sents = []

    # Заменяем "\/" на "\"
    # text = text.replace("\\/", "/")

    doc = MyDoc(text=text)
    spans = []
    # Подготовка сущностей
    for entity in entities:
        span = check_correct_token(doc,
                                    text,
                                    (entity['id'],
                                     entity['start_offset'],
                                     entity['end_offset'],
                                     entity['label'])
                                     )
        if span:
            spans.append(span)

        # if span:
        #     id2ent[entity['id']] = span

    # Отсеиваем пересекающиеся сущности
    correct_ents = span_filter(spans)

    id2ent = {span.id: span for span in correct_ents}

    for ent in correct_ents:
        write_add(file_path_log, f"{ent.id} -- {ent.text}\n")

    # Назначаем отфильтрованные сущности документу
    doc.ents = correct_ents

    # Разделение текста на предложения
    sentences = list(sentenize(text))

    # Список id сущностей по предложению
    sent2ents = {
        sentence: [entity.id for entity in correct_ents
                   if sentence.start <= entity.start < sentence.stop]
        for sentence in sentences
    }

    # Предложение по id сущности
    ent_id2sent = {
        ent_id: next((sentence.text for sentence in sentences
                      if sentence.start <= entity.start < sentence.stop), None)
        for ent_id, entity in id2ent.items()
    }

    info_sents = []
    
    # Обрабатываем предложения
    for sentence in sentences:
        sent_start, sent_end = sentence.start, sentence.stop

        # Извлекаем список ID сущностей для текущего предложения
        current_ents = sent2ents.get(sentence, [])

        # Формирование структур для предложения
        sent_doc = MyDoc(text=sentence.text, offset = sent_start)
        sent_doc.ents = [id2ent[id] for id in current_ents]

        # Поиск связей и отсутствующих связей
        sent_relations, sent_missing_relations = find_missing_relations(current_ents, relations, ent_id2sent, id2ent)


        # print("len relations:", total_rellen(relations))

        # Генерация BIO-тегов
        # bio_tags =""
        bio_tags,token_tag_dict = generate_bio_tags(sent_doc, sent_start, sent_end)

        # Добавляем информацию о текущем предложении
        info_sents.append({
            "text": (f"{sent_start}-{sent_end}", sentence.text),
            "tokens": bio_tags,
            "token_tag_dict":token_tag_dict,
            "entities_full": [(id, id2ent[id].text, f"{id2ent[id].start}-{id2ent[id].end}", id2ent[id].label) for id in current_ents],
            "entities": [(id2ent[id].text, id2ent[id].label) for id in current_ents],
            "relations": sent_relations,
            "missing_relations": sent_missing_relations,
        })

    return info_sents

def prepare_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    res_total = []

    for line in lines:
        data = json.loads(line)

        # if data['id'] in (1857, 1859):
            # Присвоение каждому предложению свои сущности и связи
        res_total.append(link_ent_sent(data))

    total_ent = 0
    total_rel = 0

    # global total_rel
    for res_sent in res_total:
        for res in res_sent:
            total_rel += len(res["relations"])
            total_ent += len(res["entities_full"])

    
    print("total_rel:", total_rel)
    print("total_ent:", total_ent)

    return res_total

# Пути (непутю)
version = "7_spironolacton"
# Входные данные
file_path = f"data\\data_bio\\data_{version}.jsonl"

# Выходные данные
file_path_relation = f"data\\data_bio\\data_relations_{version}.csv"
file_path_bio = f"data\\data_bio\\data_bio_{version}.json"
file_path_log = f"data\\data_bio\\data_log_{version}.json"

with open(file_path_log, 'w', encoding='utf-8') as file_log:
    file_log.write("")

if __name__ == "__main__":
    result = prepare_file(file_path)

    # Запись связей
    with open(file_path_relation, 'w', encoding='utf-8') as file_rel:
        file_rel.write("from$label_from$sent_from$type$to$label_to$sent_to\n")
        for sent_list in result:
            for item in sent_list:
                # total_rel_1 += len(item["relations"])
                for relation in (item["relations"]+item["missing_relations"]):
                    entity_from, label_from, text_from, relation_type, entity_to, label_to, text_to = relation
                    write_line = f"{entity_from}${label_from}${text_from}${relation_type}${entity_to}${label_to}${text_to}\n"
                    write_line = write_line.replace("•", "*")
                    write_line = write_line.replace("( ", "(")
                    file_rel.write(write_line)

    # Запись BIO
    with open(file_path_bio, 'w', encoding='utf-8') as file_bio:
        bio_list = []
        for sent in result:
            for item in sent:
               
                bio_list.append(item["token_tag_dict"])

        # Записываем весь объект в файл
        json.dump(bio_list, file_bio, ensure_ascii=False, indent=4)
        
    # Логи
    with open(file_path_log, 'w', encoding='utf-8') as file_log:
        json.dump(result, file_log, ensure_ascii=False, indent=4)

