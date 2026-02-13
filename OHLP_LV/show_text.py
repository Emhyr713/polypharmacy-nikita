import sys
sys.path.append("")

from utils.extract_text_from_pdf import extract_text_from_pdf

pdf_path = "OHLP_LV\\data\\ОХЛП_all\\1) Гликлазид _ Гликлада® _ 02.11.2022.pdf"
text = extract_text_from_pdf(pdf_path, margin_bottom=70, join_dash=True)

# Сохраняем в файл
with open("OHLP_LV\\data\\1) Гликлазид _ Гликлада® _ 02.11.2022.txt", "w", encoding="utf-8") as f:
    f.write(text)