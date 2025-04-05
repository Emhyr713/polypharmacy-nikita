import json
import os
import sys
sys.path.append("")

from export_from_pdf import extract_pdf
from utils.split_sent_razdel import split_format_text

DIR_LIST_PDF = [
    "data\\Инструкции_ГРЛС_не_вкладыши_не_сканы_1",
    "data\\Инструкции_ГРЛС_не_вкладыши_не_сканы_2"
]

def main():
    # drug_jsones = []

    # drug_jsones = extract_pdf(DIR_LIST_PDF)

    for i, dir_pdf in enumerate(DIR_LIST_PDF):
        dir_pdf_list = [dir_pdf]
        drug_jsones = extract_pdf(dir_pdf_list)
        for drug_json in drug_jsones:
            sent_list = split_format_text(drug_json['text']).split('\n')
            drug_json['sents_info'] = []
            for sent in sent_list:
                drug_json['sents_info'].append({
                    'sent':sent,
                    'tokens':[],
                    'tags':[],
                    'need_ratio':None,
                })
            drug_json['entity_list'] = []
    
        output_path = os.path.join("data_jsonl_export", f"export_info_for_test_models_{i}.json")
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(drug_jsones, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
