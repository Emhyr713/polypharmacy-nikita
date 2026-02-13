import fitz

def extract_text_from_pdf(pdf_path,
                          margin_top=0, margin_bottom=0,
                          margin_left=0, margin_right=0,
                          join_dash = True):
    with fitz.open(pdf_path) as pdf:
        full_text = ""
        
        for page in pdf:
            # Получаем размеры страницы
            page_rect = page.rect  # (x0, y0, x1, y1) — вся страница
            width = page_rect.width
            height = page_rect.height

            # Создаём прямоугольник — область для извлечения текста
            clip_rect = fitz.Rect(
                margin_left,           # x0
                margin_top,            # y0
                width - margin_right,  # x1
                height - margin_bottom # y1
            )

            # Извлекаем текст только из этой области
            page_text = page.get_text("text", clip=clip_rect)

            # Очистка и склейка строк
            filtered_lines = []
            previous_line = ""
            for line in page_text.split("\n"):
                stripped_line = line.strip()

                # if stripped_line.isdigit():
                #     continue

                if previous_line.endswith("-") and filtered_lines and join_dash:
                    filtered_lines[-1] = filtered_lines[-1][:-1] + stripped_line
                else:
                    filtered_lines.append(stripped_line)

                previous_line = stripped_line

            full_text += "\n".join(filtered_lines) + "\n"

        return full_text