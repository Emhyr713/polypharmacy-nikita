"""Модуль выдуль не посредственного вычисления."""

import csv
import logging
import pandas as pd

import numpy as np

# from drugs.models import Drug, SideEffect, DrugSideEffect

logger = logging.getLogger('fortran')

FILENAME_SIDE_E = "test_prepare_combinations\\data\\side_e_ids.csv"
FILENAME_DRUG = "test_prepare_combinations\\data\\drug_ids.csv"

def load_item2id(filename):
    # Загрузка соответствия название ЛС ↔ ID
    item2id = {}
    id2item = {}
    with open(filename, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            item2id[row["item"].strip().lower()] = int(row["id"])
            id2item[int(row["id"])] = row["item"].strip().lower()
    return item2id, id2item
    
side_e2id, id2side_e = load_item2id(FILENAME_SIDE_E)
drug2id, id2drug = load_item2id(FILENAME_DRUG)

# n_side_e = len(side_e2id)
# n_drug = len(drug2id)

# Чтение CSV-файла (предполагается, что разделитель запятая)
df = pd.read_csv("test_prepare_combinations\\data\\weights.csv", header=None, delimiter=';')

# Преобразование DataFrame в numpy array, затем в одномерный список
weight_string = df.values.flatten().tolist()

print(weight_string)


class FortranCalculator:
    """
    Вычислитель рангов для лекарств и побочных эффектов.
    """

    _DEFAULT_RANK_NAME = 'rang_base'

    @classmethod
    def get_default_rank_name(cls):
        return cls._DEFAULT_RANK_NAME

    def __init__(self, id2drug, id2side_e, weight_string):
        self.n_Drugs = len(id2drug)
        self.n_SideEffects = len(id2side_e)
        self.rangs = weight_string

    def calculate(self, nj):
        logger.debug(f"Индексы входных ЛС (nj): {nj}")

        non_zero_nj = [idx for idx in nj if idx != 0]
        unique_nj = list(set(non_zero_nj))
        num_drugs = len(unique_nj)

        # if rank_name is None:
        #     rank_name = self.get_default_rank_name()

        # logger.debug(f"Используемый ранг: {rank_name}")

        # Создаем матрицу рангов для выбранных ЛС
        # rangs = [getattr(r, rank_name) for r in DrugSideEffect.objects.all()]
        rang1 = np.zeros((num_drugs, self.n_SideEffects))

        for j, drug_idx in enumerate(unique_nj):
            for k in range(self.n_SideEffects):
                rang1[j, k] = self.rangs[self.n_SideEffects * (drug_idx - 1) + k]

        # Вычисление суммы рангов по эффектам
        rangsum = np.sum(rang1, axis=0)
        ram = np.max(rangsum)

        # Классификация
        if ram >= 1.0:
            classification = 'incompatible'
        elif ram >= 0.5:
            classification = 'caution'
        else:
            classification = 'compatible'

        context = {
            'rank_iteractions': round(float(ram), 2),
            'сompatibility_fortran': classification
        }

        # Распределение эффектов по классам
        side_effects = []
        for k in range(self.n_SideEffects):
            rank_val = rangsum[k]
            if rank_val >= 1.0:
                cls = 3
            elif rank_val >= 0.5:
                cls = 2
            else:
                cls = 1
            effect = id2side_e[k+1]
            side_effects.append({
                'se_name': effect,
                'class': cls,
                'rank': round(float(rank_val), 2)
            })

        context['side_effects'] = [
            {"сompatibility": "compatible", 'effects': []},
            {"сompatibility": "caution", 'effects': []},
            {"сompatibility": "incompatible", 'effects': []},
        ]

        side_effects.sort(key=lambda x: x['rank'], reverse=True)
        for effect in side_effects:
            cls = effect.pop('class')
            context['side_effects'][cls - 1]['effects'].append(effect)

        logger.debug(f'len(side_effects) = {len(side_effects)}')

        # Анализ потенциальных ЛС
        rangs_matrix = np.array(self.rangs).reshape(self.n_Drugs, self.n_SideEffects)
        unique_nj_sub_1 = [idx - 1 for idx in unique_nj]
        drugs_class_2, drugs_class_3 = [], []

        logger.debug(f'rangs_matrix = {rangs_matrix}')
        logger.debug(f'unique_nj_sub_1 = {unique_nj_sub_1}')
        for j in range(self.n_Drugs):
            if j not in unique_nj_sub_1:
                new_rangsum = rangsum + rangs_matrix[j]
                max_rang = np.max(new_rangsum)
                logger.debug(f'j = {j}')
                logger.debug(f'max_rang = {max_rang}')
                if max_rang >= 1.0:
                    drugs_class_3.append(j)
                elif max_rang >= 0.5:
                    drugs_class_2.append(j)

        drug_array2 = [{'name': id2drug[j+1], 'class': 2} for j in drugs_class_2]
        drug_array3 = [{'name': id2drug[j+1], 'class': 3} for j in drugs_class_3]

        context['combinations'] = [
            {"сompatibility": 'cause', "drugs": [d['name'] for d in drug_array2]},
            {"сompatibility": 'incompatible', "drugs": [d['name'] for d in drug_array3]},
        ]

        context['drugs'] = [id2drug[i] for i in unique_nj]

        return context
    
if __name__ == "__main__":
    calc = FortranCalculator(id2drug, id2side_e, weight_string)

    print(calc.calculate([1,2]))

