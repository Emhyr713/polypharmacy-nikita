def translate(translations, text):
    """Переводит текст по словарю, если есть соответствие."""
    translated = translations.get(text.lower(), None)
    if translated:
        return translated
    else:
        print(f"Нет в словаре: '{text}'")
        return text