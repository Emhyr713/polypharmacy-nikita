import os
import json
import logging
import sqlite3

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIR_DATA = "compute_object_p\\data"

CONVERT_RATE = {
        "Очень часто": 0.55,
        "Часто": 0.055,
        "Нечасто": 0.0055,
        "Редко": 0.00055,
        "Очень редко": 0.00055,
        "Частота неизвестна": 0.000055,
    }

class ComputeObjectF:
    """
    Класс для расчёта ошибки свёртки по данным о ЛС и побочных эффектах.
    """
    def __init__(self, drug_table, drug_graph, drug_side_e_dataset):
        self.drug_table = drug_table
        self.drug_graph = drug_graph
        self.drug_side_e_dataset = drug_side_e_dataset

        # Быстрый доступ к узлу по id
        self.node_map_graph = {node["id"]: node for node in self.drug_graph["nodes"]}
        self.node_map_table = {node["node_id"]: node["convolution_table"]["1"] for node in self.drug_table["tables"]}

        # Список ЛС из графа (названия приводятся к нижнему регистру)
        self.drug_list = self._get_drug_list()

        # Словари побочных эффектов для каждого ЛС из датасета и графа 
        self.side_effect_dataset = self._get_side_effect_freq_dataset()
        self.side_effect_table = self._find_side_e_by_drug()

    def _get_label(self, node_id):
        """Возвращает метку узла по его id."""
        return self.node_map_graph[node_id].get("label")

    def _get_drug_list(self):
        """Извлекает список лекарственных средств (prepare) из графа."""
        drugs = []
        for node in self.drug_graph.get("nodes", []):
            if node.get("label") == "prepare":
                if all(self._get_label(parent) == "group" for parent in node.get("parents", [])):
                    drugs.append((node["id"], node["name"].lower()))
        return drugs
    
    def _find_side_e_by_drug(self):
        """
        Находит все узлы с label "side_e", которые достижимы из узла с заданным start_id.
        Достижимость определяется по направленным связям: если id узла присутствует в поле
        "parents" другого узла, то последний считается дочерним для первого.
        """
        drug_side_effect = {}

        for drug_id, drug_name in self.drug_list:
            # Построим отображение: родитель -> список дочерних узлов.
            children = {}
            for node in self.drug_graph.get("nodes", []):
                for p in node.get("parents", []):
                    children.setdefault(p, []).append(node["id"])

            # Поиск всех достижимых узлов с помощью обхода в глубину (DFS)
            reachable = set()
            stack = [drug_id]
            while stack:
                current = stack.pop()
                if current in reachable:
                    continue
                reachable.add(current)
                # Добавляем всех детей текущего узла, если они есть
                for child in children.get(current, []):
                    if child not in reachable:
                        stack.append(child)

            # Фильтруем только те узлы, которые имеют label "side_e"
            drug_side_effect[drug_name] = {
                node["name"]:self._get_convolution_probability(node["id"])
                for node in self.drug_graph.get("nodes", [])
                if node["id"] in reachable and node.get("label") == "side_e"
            }

        return drug_side_effect
    
    def _get_side_effect_freq_dataset(self):
        """
        Получает датасет побочных эффектов для каждого ЛС из JSON.
        Возвращает словарь: ключ - название препарата (нижний регистр), значение - список побочных эффектов.
        """
        return {name:self.drug_side_e_dataset[name]['side_e_parts'] for id, name in self.drug_list}


    def _get_convolution_probability(self, node_id):
        """Возвращает вероятность свёртки по id узла из таблицы."""
        return self.node_map_table[node_id]

    def calculate(self):
        """
        Вычисляет суммарную ошибку (loss) для каждого ЛС.
        Возвращает словарь с данными о суммарной ошибке, найденных и отсутствующих побочных эффектах.
        """
        result = {}
        for drug_id, drug in self.drug_list:

            # Получаем побочные эффекты для препарата, если они представлены списком, преобразуем в словарь
            effects_dataset = self.side_effect_dataset.get(drug, {})
            effects_table = self.side_effect_table.get(drug, {})

            print(f"{drug} dataset: {effects_dataset}")
            print()
            print(f"{drug} table: {effects_table}")

            # Общие побочные эффекты между датасетом и таблицей
            common = [
                (key, effects_dataset.get(key), effects_table.get(key))
                for key in effects_table if key in effects_dataset
            ]
            # Побочные эффекты, отсутствующие в датасете
            missing = [key for key in effects_table if key not in effects_dataset]

            if common:
                try:
                    loss = sum((CONVERT_RATE.get(val, 0) - cur) ** 2 for _, val, cur in common)
                except Exception as e:
                    logger.error("Ошибка расчёта для %s: %s", drug, e)
                    loss = None
            else:
                loss = None

            result[drug] = {"summ_loss": loss,
                            "finded_len": len(common),
                            "missing_len":len(missing),
                            "finded": common,
                            "missed": missing}
        return result


if __name__ == "__main__":
    # Загрузка данных из JSON файлов
    with open(f"{DIR_DATA}\\merged_graph_fozinopril_ramipril_Tables.json", "r", encoding="utf-8") as table_file:
        drug_table_data = json.load(table_file)

    with open(f"{DIR_DATA}\\merged_graph_fozinopril_ramipril_Parse.json", "r", encoding="utf-8") as graph_file:
        drug_graph_data = json.load(graph_file)

    with open(f"{DIR_DATA}\\sef_dataset.json", "r", encoding="utf-8") as dataset_file:
        drug_side_e_dataset = json.load(dataset_file)

    compute_obj = ComputeObjectF(drug_table_data, drug_graph_data, drug_side_e_dataset)
    result_compute = compute_obj.calculate()

    with open(f"{DIR_DATA}\\result_compute.json", "w", encoding="utf-8") as newfile:
        json.dump(result_compute, newfile, ensure_ascii=False, indent=4)