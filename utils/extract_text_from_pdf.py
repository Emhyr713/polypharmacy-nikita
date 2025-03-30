import fitz

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        text = ""
        previous_line = ""
        
        filtered_lines = []
        for page in pdf:
            page_text = page.get_text("text")
            
            # f.write(f"{pdf_path}: {page_text}")
 
            for line in page_text.split("\n"):
                stripped_line = line.strip()

                if not stripped_line.isdigit() and len(stripped_line) >= 3:
                    if previous_line.endswith("-") and filtered_lines:
                        # Соединяем перенос слова, если есть предыдущая строка
                        filtered_lines[-1] = filtered_lines[-1][:-1] + stripped_line
                    else:
                        filtered_lines.append(stripped_line)
                    previous_line = stripped_line
            
        text += "\n".join(filtered_lines) + "\n"
    
    return text