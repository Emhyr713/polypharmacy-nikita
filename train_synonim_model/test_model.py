from sentence_transformers import SentenceTransformer, util

# путь к сохранённой модели
my_model = SentenceTransformer('train_synonim_model\\data\\synonym-model_3')
# base_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# пара слов
pairs = [
    ("альфа-адреномиметическое действие", "α-адреномиметическое действие"),
    ("β-адреномиметическое действие", "α-адреномиметическое действие"),
    ("бета-адреномиметическое действие", "альфа-адреномиметическое действие"),
    ("бета-адреномиметическое действие", "β-адреномиметическое действие")
]

print("my_model")


for a, b in pairs:
    emb_a = my_model.encode(a, convert_to_tensor=True)
    emb_b = my_model.encode(b, convert_to_tensor=True)
    sim_trained = util.cos_sim(emb_a, emb_b).item()

    # emb_a = base_model.encode(a, convert_to_tensor=True)
    # emb_b = base_model.encode(b, convert_to_tensor=True)
    # sim_base = util.cos_sim(emb_a, emb_b).item()

    print(f"trained_model: {a} ~ {b} → train similarity = {sim_trained:.3f}")


