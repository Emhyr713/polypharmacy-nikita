import json
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from process_dataset import extract_synonym_pairs_by_cluster
from torch.utils.data import DataLoader

# Фиксация состояния генераторов случайных чисел для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Пути к данным
SYNONYM_DATASET_FILENAME = "train_synonim_model/data/synonim_dataset.json"
TEXT_DATASET_FILENAME = "collect_text_corpus/data/text_corpus_grls_rlsnet.txt"

# Загрузка кластерного синонимичного датасета
with open(SYNONYM_DATASET_FILENAME, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Извлечение всех синонимичных пар
pairs_by_cluster = extract_synonym_pairs_by_cluster(dataset["clusters"])
synonym_pairs = [pair for cluster in pairs_by_cluster.values() for pair in cluster]

# Формирование перечня всех уникальных слов
all_words = list({w for pair in synonym_pairs for w in pair})

# Генерация негативных примеров случайным образом
negative_pairs = set()
while len(negative_pairs) < len(synonym_pairs):
    a, b = random.sample(all_words, 2)
    if (a, b) not in synonym_pairs and (b, a) not in synonym_pairs:
        negative_pairs.add((a, b))
negative_pairs = list(negative_pairs)

# Объединение положительных и отрицательных примеров с метками
examples = []
for w1, w2 in synonym_pairs:
    examples.append(InputExample(texts=[w1, w2], label=1.0))
for w1, w2 in negative_pairs:
    examples.append(InputExample(texts=[w1, w2], label=0.0))

# Перемешивание и разбиение на train/val/test
import random

# Допустим, examples уже сформирован и содержит InputExample с метками 0.0/1.0
random.shuffle(examples)

n_total = len(examples)
n_train = int(0.8 * n_total)
n_val   = int(0.1 * n_total)

# оставшиеся примеры автоматически пойдут в тестовую выборку
n_test  = n_total - n_train - n_val

train_examples = examples[:n_train]
val_examples   = examples[n_train:n_train + n_val]
test_examples  = examples[n_train + n_val:]

train_dataloader = DataLoader(train_examples, shuffle=True,  batch_size=32)
val_dataloader   = DataLoader(val_examples,   shuffle=False, batch_size=32)
test_dataloader  = DataLoader(test_examples,  shuffle=False, batch_size=32)


# Инициализация модели
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Обучение модели на синонимичных и негативных парах
train_loss = losses.CosineSimilarityLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100,
    output_path='train_synonim_model/data/synonym-model'
)

# Функция вычисления метрик на выборке
def evaluate_on_dataloader(dataloader, threshold=0.5):
    true_labels = []
    pred_scores = []
    for batch in dataloader:
        embeddings = model.encode(batch.texts, convert_to_numpy=True)
        # embeddings содержит массив shape (2*batch_size, dim), нужно попарно
        half = embeddings.shape[0] // 2
        emb1, emb2 = embeddings[:half], embeddings[half:]
        cos_sims = np.sum(emb1 * emb2, axis=1) / (
            np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
        )
        pred_scores.extend(cos_sims.tolist())
        true_labels.extend(batch.label.numpy().tolist())
    # Классификация по порогу
    preds = [1 if s >= threshold else 0 for s in pred_scores]
    metrics = {
        "accuracy": accuracy_score(true_labels, preds),
        "precision": precision_score(true_labels, preds),
        "recall": recall_score(true_labels, preds),
        "f1": f1_score(true_labels, preds),
        "roc_auc": roc_auc_score(true_labels, pred_scores)
    }
    return metrics

# Оценка на валидационной выборке (для контроля обучения)
val_metrics = evaluate_on_dataloader(val_dataloader)
print("Validation metrics:", val_metrics)

# Итоговая оценка на тестовой выборке
test_metrics = evaluate_on_dataloader(test_dataloader)
print("Test metrics:", test_metrics)
