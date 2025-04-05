make_side_effect_dataset:
Парсит информацию из ГРЛС, rlsnet.ru, drugs.com. ГРЛС инструкции берутся из .pdf файлов. Сначала побочки берутся из ГРЛС, если нет ссылки, то из rlsnet.ru, иначе из drugs.com.
    data:
        drugs_table4.csv -- Хранит ссылки на ГРЛС, rlsnet.ru, drugs.com.
        side_effects_dict_translation.json -- словарь для переводов.
        list_class_side_e_edit.txt -- список глав побочек.
        side_dataset.json -- Побочки с главами и ссылками на ресурс.
        sef_dataset.json -- ЛС: ПЭ - Частота.
        