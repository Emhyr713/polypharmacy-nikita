
import sys
sys.path.append("")

from create_graph.KnowledgeGraphExplorer import KnowledgeGraphExplorer
from create_graph.NormalizeListEdges import NormalizeListEdges
from create_graph.SemanticEmbeddingProcessor import SemanticEmbeddingProcessor


def simplification_graph(edges_df, model_path, treshold=0.98):
    emb_processor = SemanticEmbeddingProcessor(model_path=model_path)
    normalizer = NormalizeListEdges(embedding_processor=emb_processor)
    explorer = KnowledgeGraphExplorer(embedding_processor=emb_processor)

    # Нормализация
    normalized_df = normalizer.normalize_list_edges(
        df_edges=edges_df, similarity_threshold=treshold
    )

    # Построение графа и поиск
    explorer.build_graph(normalized_df)

    return explorer.graph