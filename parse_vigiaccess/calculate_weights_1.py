import pandas as pd
from typing import Dict, Tuple, List

class SideEffectAnalyzer:
    def __init__(self, blank_file: str, dataset_file: str, output_file: str):
        """
        Инициализация анализатора побочных эффектов.

        :param blank_file: Путь к Excel-файлу с шаблонами
        (лекарства, побочные эффекты, матрица, общее).
        :param dataset_file: Путь к Excel-файлу с выгрузкой VigiAccess.
        :param output_file: Путь для сохранения результатов обработки.
        """
        self.blank_file = blank_file
        self.dataset_file = dataset_file
        self.output_file = output_file

        self.df_blank_side_e = pd.read_excel(self.blank_file, sheet_name='Side_e', header=None)
        self.df_matrix = pd.read_excel(self.blank_file, sheet_name='Common', header=None)
        self.df_blank_total = pd.read_excel(self.blank_file, sheet_name='Total', header=None)
        self.df_blank_drugs = pd.read_excel(self.blank_file, sheet_name='Drugs', header=None)
        self.df_dataset = pd.read_excel(self.dataset_file, sheet_name='Sheet', header=None)

        self.side_dict, self.se_list_ru = self._load_side_effects()
        self.drug_list = self._get_drugs_list()
        self.drug_data_dict = self._get_vigi_data()

    def _load_side_effects(self) -> Tuple[Dict[str, float], pd.Series]:
        """
        Загрузка списка побочных эффектов и их рангов из шаблона.

        :return: Словарь {название_побочного_эффекта: ранг},
        а также список побочных эффектов на русском.
        """
        se_rus = self.df_blank_side_e.iloc[1:, 1]
        se_eng = self.df_blank_side_e.iloc[1:, 2]
        ranks = self.df_blank_side_e.iloc[1:, 3]

        side_dict = {
            eng.lower().strip(): rank
            for rus, eng, rank in zip(se_rus, se_eng, ranks)
            if pd.notna(rus) and pd.notna(eng) and pd.notna(rank)
        }

        return side_dict, se_rus.dropna().reset_index(drop=True)
    
    def _get_drugs_list(self) -> pd.Series:
        """
        Получение списка лекарств из шаблона.

        :return: Серия с названиями лекарств.
        """
        return self.df_blank_drugs.iloc[1:, 1].str.strip().str.lower()

    def _get_vigi_data(self) -> Dict[str, dict]:
        """
        Формирование словаря частот побочных эффектов по каждому лекарству.

        :return: Словарь вида:
                 {
                     'название_лекарства': {
                         'total': общее_число_обращений,
                         'data': {побочный_эффект: частота}
                     }
                 }
        """
        drug_vigi_rus = self.df_dataset.iloc[0, 1:].str.strip().str.lower()
        total_requests = self.df_dataset.iloc[2, 1:]
        side_e_eng = self.df_dataset.iloc[17:, 0]

        drug_data_dict = {}
        for i, (drug, total_request) in enumerate(zip(drug_vigi_rus, total_requests)):
            if drug in self.drug_list.values:
                freq_list = self.df_dataset.iloc[17:, i + 1]
                drug_data_dict[drug] = {
                    'total': total_request if pd.notna(total_request) else 0,
                    'data': {
                        side_e.lower().strip(): freq
                        for side_e, freq in zip(side_e_eng, freq_list)
                        if pd.notna(side_e) and side_e.lower()
                    }
                }
        return drug_data_dict

    def _create_matrix(self) -> pd.DataFrame:
        """
        Формирование матрицы побочных эффектов (основа для результата).

        :return: DataFrame с номерами и названиями ПЭ
        по столбцам и лекарствами по строкам.
        """
        num_effects = len(self.se_list_ru)
        row0 = [None, None] + list(range(1, num_effects + 1))
        row1 = [None, 'ЛС/ПЭ'] + self.se_list_ru.tolist()
        data_rows = [[i + 1, drug] + [None] * num_effects for i, drug in enumerate(self.drug_list)]
        return pd.DataFrame([row0, row1] + data_rows)

    def _write_total_counts(self):
        """
        Запись общего количества обращений
        по каждому препарату в шаблон `Total`.
        """
        for row_idx in range(1, len(self.df_blank_total)):
            drug_name = self.df_blank_total.iloc[row_idx, 1].lower().strip()
            if drug_name in self.drug_data_dict:
                self.df_blank_total.iloc[row_idx, 2] = self.drug_data_dict[drug_name]['total']

    def _calculate_weights(self, df_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет весов побочных эффектов на основе частоты и ранга.

        Формула: weight = round((freq * rank / total) ** (2/5), 2)

        :param df_matrix: Матрица с названиями ЛС и ПЭ.
        :return: Обновленная матрица с рассчитанными весами.
        """
        start_row = 2
        start_column = 2

        for row_idx in range(start_row, len(df_matrix)):
            drug_name = df_matrix.iloc[row_idx, 1].lower().strip()
            drug_data = self.drug_data_dict.get(drug_name)
            
            if not drug_data:
                df_matrix.iloc[row_idx, start_column:] = 0
                continue

            total = drug_data['total']
            freq_data = drug_data['data']
            
            for col_idx in range(len(self.side_dict)+1):
                side_e_eng = self.df_blank_side_e.iloc[col_idx + 1, 2]
                if pd.isna(side_e_eng):
                    continue
                side_e_eng = side_e_eng.lower().strip()

                if '?' in side_e_eng or side_e_eng not in self.side_dict:
                    df_matrix.iat[row_idx, col_idx + 2] = 0
                    continue

                rank = self.side_dict[side_e_eng]
                freq = freq_data.get(side_e_eng, 0)
                weight = round((freq * rank / total)**(2/5), 2) if freq and total else 0
                df_matrix.iat[row_idx, col_idx + 2] = weight

        return df_matrix

    def _save_results(self, df_result: pd.DataFrame):
        """
        Сохранение результатов обработки в Excel-файл.

        :param df_result: Обновленная матрица с рассчитанными весами.
        """
        with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
            df_result.to_excel(writer, sheet_name='Common', index=False, header=False)
            self.df_blank_total.to_excel(writer, sheet_name='Total', index=False, header=False)
            self.df_blank_side_e.to_excel(writer, sheet_name='Side_e', index=False, header=False)
            self.df_blank_drugs.to_excel(writer, sheet_name='Drugs', index=False, header=False)

    def run(self):
        """
        Основной метод запуска анализа: строит матрицу, записывает общее количество,
        рассчитывает веса и сохраняет результаты.
        """
        df_matrix = self._create_matrix()
        self._write_total_counts()
        df_result = self._calculate_weights(df_matrix)
        self._save_results(df_result)

if __name__ == "__main__":
    analyzer = SideEffectAnalyzer(
        blank_file='parse_vigiaccess/data/blank.xlsx',
        dataset_file='parse_vigiaccess/data/Vigiaccess_20250515_000505.xlsx',
        output_file='parse_vigiaccess/data/Список побочных эффектов edit_2.xlsx'
    )
    analyzer.run()
