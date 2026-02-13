import json
import csv
import sys
sys.path.append("")

from sentence_transformers import SentenceTransformer
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

# Путь к сохранённой модели
TERMS_FILE = "visualization_embedding\\data\\terms_list_verified_folder.json"
with open(TERMS_FILE, 'r', encoding='utf-8') as file:
    term_list = json.load(file)
terms = [term[0] for term in term_list]

# Загрузка модели
print("Загрузка модели...")
model = SentenceTransformer('train_synonim_model\\data\\synonym-model_1')

# Генерация эмбеддингов
print("Генерация эмбеддингов для терминов...")
embeddings = model.encode(terms, show_progress_bar=True)

# Получение всех уникальных пар
pairs = list(combinations(terms, 2))
print(f"Найдено {len(pairs)} уникальных пар. Вычисление косинусного сходства...")

# Список для хранения результатов
results = []

# Обработка пар с прогресс-баром
for term1, term2 in tqdm(pairs, desc="Обработка пар", unit="пара"):
    idx1 = terms.index(term1)
    idx2 = terms.index(term2)
    emb1 = embeddings[idx1].reshape(1, -1)
    emb2 = embeddings[idx2].reshape(1, -1)
    cos_sim = cosine_similarity(emb1, emb2)[0][0]
    round(cos_sim, 4)
    results.append({
        'term1': term1,
        'term2': term2,
        'cosine_similarity': float(cos_sim)
    })

results.sort(key=lambda x: x['cosine_similarity'], reverse=True)

# Сохранение в CSV
csv_file = 'visualization_embedding\\data\\similarity_pairs_2.csv'
with open(csv_file, mode='w', encoding='utf-8', newline='',) as file:
    writer = csv.DictWriter(file, fieldnames=['term1', 'term2', 'cosine_similarity'])
    writer.writeheader()
    writer.writerows(results)

print(f"\nРезультаты сохранены в файл: {csv_file}")