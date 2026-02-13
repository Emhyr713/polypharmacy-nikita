from sentence_transformers import SentenceTransformer

def get_embeddings(terms, path):
    # Загрузка модели
    my_model = SentenceTransformer(path)

    # Вычисление эмбеддингов
    embeddings = my_model.encode(terms, convert_to_tensor=False)
    return embeddings