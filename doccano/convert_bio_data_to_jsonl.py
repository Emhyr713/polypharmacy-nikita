import json
import sys
sys.path.append("")

from convertors.umf2jsonl import umf2jsonl
from convertors.BIO2umf import BIO2umf_d

FILENAME = "doccano\\data\\labeled_data_dlc.json"
with open(FILENAME, "r", encoding="utf-8") as file:
    dataset = json.load(file)

OUTPUT_FILENAME = "doccano\\data\\labeled_data_dlc.jsonl"
with open(OUTPUT_FILENAME, "w", encoding="utf-8") as file:
    for drug in dataset:
        for key, data in drug.items():
            if data:
                # Convert the dictionary to a JSON string before writing
                json_line = json.dumps(umf2jsonl(BIO2umf_d(data)), ensure_ascii=False)
                file.write(json_line)
                file.write("\n")