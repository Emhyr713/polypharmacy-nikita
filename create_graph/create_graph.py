import pandas as pd
import networkx as nx
from pathlib import Path
import json

from KnowledgeGraphExplorer import KnowledgeGraphExplorer
from NormalizeListEdges import NormalizeListEdges
from SemanticEmbeddingProcessor import SemanticEmbeddingProcessor

import sys
sys.path.append("")
from utils.logger import get_logger
from process_yEd_graph.process_nx_graph import ProcessNxGraph

logger = get_logger(__name__)


def read_seed_words(filename: str) -> list[str]:
    """
    Считывает список начальных слов из текстового файла.
    Каждое слово/фраза на новой строке.
    Пустые строки игнорируются.
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл с начальными словами не найден: {filename}")

    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    if not words:
        raise ValueError(f"Файл {filename} пуст или содержит только пустые строки.")

    logger.info(f"Загружено {len(words)} начальных слов из {filename}")
    return words

def sanitize_graph_attributes(graph):
    """
    Приводит все атрибуты графа к типам, поддерживаемым GEXF.
    Заменяет tuple, list, dict, None и др. на строки.
    """
    def make_serializable(value):
        if value is None:
            return "null"
        elif isinstance(value, (list, tuple)):
            return "; ".join(map(str, value))
        elif isinstance(value, dict):
            return "; ".join([f"{k}: {v}" for k, v in value.items()])
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)  # все остальные типы — в строку

    # Очищаем атрибуты узлов
    for node, attr in graph.nodes(data=True):
        for key in list(attr.keys()):
            attr[key] = make_serializable(attr[key])

    # Очищаем атрибуты рёбер
    for u, v, attr in graph.edges(data=True):
        for key in list(attr.keys()):
            attr[key] = make_serializable(attr[key])

    # Разобраться с A->A, A<->B


    return graph

def main():
    LINKS_FILENAME = "process_yEd_graph/data/list_edges_verified_folder.csv"
    MODEL_PATH = "train_synonim_model/data/synonym-model_1"
    OUTPUT_DIR = "create_graph/data/graph"
    SEED_FILENAME = "create_graph/data/seed_words.txt"
    ABBR_SYNONYM_FILENAME = "data/dictonary_synonims_simple.json"

    sim_threshold = 0.96

    logger.info("Запуск обработки графа знаний")

    # --- Инициализация ---
    try:
        emb_processor = SemanticEmbeddingProcessor(
            model_path=MODEL_PATH, abbr_dataset_path=ABBR_SYNONYM_FILENAME
        )
        edges_df = pd.read_csv(LINKS_FILENAME, sep=";", dtype=str).fillna("")
        normalizer = NormalizeListEdges(embedding_processor=emb_processor)
        explorer = KnowledgeGraphExplorer(embedding_processor=emb_processor)

        logger.info("Модель и компоненты загружены.")
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        return

    # --- Цикл команд ---
    logger.info("Ожидание: 'run <name>', 'threshold <0..1>', 'exit'")

    while True:
        try:
            cmd = input("\n> ").strip().lower()
            if not cmd:
                continue

            # Выход
            if cmd == "exit":
                logger.info("Завершение работы...")
                break

            # Изменение порога
            elif cmd.startswith("threshold "):
                try:
                    new_threshold = float(cmd.split(maxsplit=1)[1])
                    if 0 < new_threshold < 1:
                        sim_threshold = new_threshold
                        logger.info(f"Порог схожести изменён: {new_threshold}")
                    else:
                        logger.warning("Порог должен быть в диапазоне (0, 1)")
                except (IndexError, ValueError):
                    logger.warning("Использование: threshold <число>")

            # Запуск построения графа
            elif cmd.startswith("run "):
                filename = cmd.split(maxsplit=1)[1].strip()
                if not filename:
                    logger.warning("Укажите имя файла: run <имя>")
                    continue

                output_path = f"{OUTPUT_DIR}/{filename}.gexf"

                try:
                    # Перечитываем seed-слова
                    seed_words = read_seed_words(SEED_FILENAME)

                    # Нормализация
                    threshold = sim_threshold
                    normalized_df = normalizer.normalize_list_edges(
                        df_edges=edges_df, similarity_threshold=threshold
                    )
                    logger.info(f"Нормализовано рёбер: {len(normalized_df)}")

                    # Построение графа и поиск
                    explorer.build_graph(normalized_df)
                    subgraph = explorer.explore_from_seeds(
                        seed_words=seed_words, similarity_threshold=threshold
                    )

                    if subgraph.number_of_nodes() == 0:
                        logger.warning("Подграф пуст.")
                    else:
                        subgraph = sanitize_graph_attributes(subgraph)
                        nx.write_gexf(subgraph, output_path)
                        ProcessNxGraph().check_graph_cycles(subgraph)
                        logger.info(f"Граф сохранён: {output_path}")

                except Exception as e:
                    logger.error(f"Ошибка при построении графа: {e}")

            else:
                logger.warning("Доступные команды: "
                               "run <name>, "
                               "threshold <val>, "
                               "exit")

        except KeyboardInterrupt:
            logger.info("\nПрервано пользователем.")
            break
        except Exception as e:
            logger.error(f"Ошибка обработки команды: {e}")

    logger.info("Работа завершена.")


if __name__ == "__main__":
    main()