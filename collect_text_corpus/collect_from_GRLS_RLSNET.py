import json
import sys
sys.path.append("")

from utils.split_sent_razdel import split_format_text

TEXT_GRLS_FILENAME = "extract_text_from_instructions\\data\\extracted_data_all.json"
TEXT_RLSNET_FILENAME = "text_corpus_addiction\\data\\rlsnet_texts.json"

SAVE_FILENAME = "collect_text_corpus\\data\\text_corpus_grls_rlsnet.txt"

text_corpus = ""

# Извлечение текста
with open(TEXT_GRLS_FILENAME, "r", encoding="utf-8") as file:
    dataset_grls = json.load(file)
with open(TEXT_RLSNET_FILENAME, "r", encoding="utf-8") as file:
    dataset_rlsnet = json.load(file)

for item in dataset_grls:
    text_corpus+=split_format_text(item["text"])

for item in dataset_rlsnet:
    for sent_list in item.values():
        if sent_list:
            for sent in sent_list:
                text_corpus+=sent+'\n'

with open(SAVE_FILENAME, 'w', encoding='utf-8') as f:
    f.write(text_corpus) 

