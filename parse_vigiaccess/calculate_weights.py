import pandas as pd

def preprocess_side_e_dict(df):
    """Создание словаря для побочных эффектов"""
    se_rus = df.iloc[0, 2:]  # ПЭ рус (начиная с третьего столбца)
    se_eng = df.iloc[1, 2:]  # SE eng (начиная с третьего столбца)
    ranks = df.iloc[2, 2:]   # Ранги (начиная с третьего столбца)

    side_dict = {}
    for rus, eng, rank in zip(se_rus, se_eng, ranks):
        if pd.notna(rus) and pd.notna(eng) and pd.notna(rank):
            side_dict[eng.lower().strip()] = rank

    return side_dict

def get_freq_side_by_drug(df):
    """Формирование словаря частот каждой побочки по каждому лекарству"""
    drug_rus = df.iloc[0, 1:]       # Название ЛС на русском
    total_requests = df.iloc[2, 1:] # Общее количество обращений

    se_eng = df.iloc[17:, 0]        # Список побочек на английском
    drug_dict = {}
    for i, (drug, total_request) in enumerate(zip(drug_rus, total_requests)):
        if pd.notna(drug) and pd.notna(total_request):
            freq_list = df.iloc[17:, i+1]
            se2freq = {side_e.lower():freq for side_e, freq in zip(se_eng, freq_list)}
            drug_dict[drug.lower().strip()]={'total':total_request, 
                             'data':se2freq
                             }
    return drug_dict

def calculate_weights(df_blank, side_dict, drug_dict):
    print("drug_dict", sorted(drug_dict.keys()))
    for row_idx in range(len(df_blank)):
        if row_idx < 3:  # Пропускаем заголовки
            continue
        
        drug_name = df_blank.iloc[row_idx, 1].lower().strip()  # Название лекарства
        current_drug_dict = drug_dict.get(drug_name, None)
        if current_drug_dict is None:
            print("df_blank.iloc[row_idx, 1]:", df_blank.iloc[row_idx, 1], "drug_name:", drug_name, "current_drug_dict:", current_drug_dict)
            continue

        total = current_drug_dict['total']
        drug_data = current_drug_dict['data']

        df_blank.at[row_idx, 2] = total
        
        # Начинаем с 3-го столбца (C)
        for col_idx in range(3, len(df_blank.columns)):  
            side_e_eng = df_blank.iloc[1, col_idx].lower().strip()
            if '?' in side_e_eng:
                df_blank.at[row_idx, col_idx] = "???"
                continue

            rank = side_dict[side_e_eng]
            freq = drug_data.get(side_e_eng, "???") 
            if '?' in str(freq):
                df_blank.at[row_idx, col_idx] = '???'
            else:
                df_blank.at[row_idx, col_idx] = freq * rank / total

    return df_blank

if __name__ == "__main__":
    # Чтение данных из Excel-файла
    file_path = 'parse_vigiaccess\\data\\Список побочных эффектов 2.xlsx'
    df_blank = pd.read_excel(file_path, sheet_name='NEW TABLE', header=None)

    file_path_dataset = "parse_vigiaccess\\data\\Vigiaccess_20250515_000505.xlsx"
    df_dataset = pd.read_excel(file_path_dataset, sheet_name='Sheet', header=None)

    side_e_dict = preprocess_side_e_dict(df_blank)
    drug_dict = get_freq_side_by_drug(df_dataset)
    # print("drug_dict:", drug_dict)

    df_result = calculate_weights(df_blank, side_e_dict, drug_dict)

    # Сохраняем обратно в Excel
    file_path_res = 'parse_vigiaccess\\data\\Список побочных эффектов edit.xlsx'
    df_result.to_excel(file_path_res, index=False, header=False)