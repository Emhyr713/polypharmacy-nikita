import re

def remove_brackets_deep(text):
    stack = []
    result = ""
    i = 0
    while i < len(text):
        if text[i] in "[({":            # Открывающие скобки
            stack.append(text[i])
        elif text[i] in "]})":          # Закрывающие скобки
            if stack:
                stack.pop()
        elif not stack:                 # Если стек пуст, добавляем символ в результат
            result += text[i]
        i += 1
    return result

def remove_brackets(text):
    # Удаление круглых, квадратных и фигурных скобок с содержимым, включая \n
    text = re.sub(r'\(.*?\)', '', text, flags=re.DOTALL)  # Удаляет круглые скобки
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)  # Удаляет квадратные скобки
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)  # Удаляет фигурные скобки
    return text
