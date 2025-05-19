import json
import sys
sys.path.append("")

from utils.split_sent_razdel import split_format_text

EMB_SIDE_E_JSON = "visualization_embedding\\data\\embedding_blank_nodes2_med_embeddings_model_distiluse_base_multilingual.json"

with open(EMB_SIDE_E_JSON, "r", encoding = "utf-8") as file:
    text = json.load(file)["text"]


print("text:", split_format_text(text).split("\n"))
print("len text:", len(split_format_text(text).split("\n")))
