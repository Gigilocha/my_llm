import re
import unicodedata


# Ссылки
RE_URL = re.compile(
    r'(?:'
    r'https?://\S+|'
    r'ftp://\S+|'
    r'www\.\S+|'
    r'\b[\w.-]+\.(?:com|ru|org|net|io|edu|gov|uk|de|fr|jp|cn|ai|dev|app)\b(?:/\S*)?|'
    r'<https?://[^>]+>|'
    r'\[([^\]]+)\]\(https?://[^)]+\)'   # markdown-ссылки
    r')',
    re.IGNORECASE
)

RE_EMPTY_BRACKETS = re.compile(r'(\(\s*\)|\[\s*\]|\{\s*\}|<\s*>)')
RE_EMPTY_QUOTES   = re.compile(r'(["\']\s*["\']|«\s*»|„\s*"|‹\s*›|‘\s*’|“\s*”)')

RE_PUNCT_REPEAT = re.compile(r'([.,!?;:\-—–=*_~])\1{3,}')  # >3 подряд
RE_WS           = re.compile(r'[ \t\u00A0\u2007\u202F\u2009\u200A\u2002\u2003]+')
RE_NEWLINES     = re.compile(r'\n{3,}')
RE_SPACE_NL     = re.compile(r'[ \t]*\n[ \t]*')

# Управляющие символы (кроме \n и \t, их обработаем отдельно)
RE_CONTROL = re.compile(
    r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F'
    r'\u200B-\u200F\u2028-\u202F\u2060-\u206F\uFEFF\u00AD]'
)


# Очистка текст
# Если предложение получается пустым, то возвращаем None, в обычном случае возвращаем очищенное преложение
def clean_text(text: str, min_len: int, max_len: int, max_punct: int = 3) -> str | None:
    if not isinstance(text, str):
        return None

    # Нормализация Unicode
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text)                 
    
    # Удаление ссылок
    text = RE_URL.sub(" ", text)

    # Удаление пустых кавычек
    prev = None
    while prev != text:
        prev = text
        text = RE_EMPTY_QUOTES.sub("", text)
        # на случай "" с пробелом внутри → "" → пусто
        text = re.sub(r'["\']\s*["\']', '', text)

    # Удаление пустых скобок 
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = RE_EMPTY_BRACKETS.sub("", text)
        # часто остаётся «( )» с пробелом внутри
        text = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', text)

    # Повторяющиеся пунктуация
    text = RE_PUNCT_REPEAT.sub(lambda m: m.group(1) * max_punct, text)

    # Пробелы и переносы
    text = RE_SPACE_NL.sub('\n', text)              # пробелы вокруг \n
    text = RE_WS.sub(' ', text)                     # множественные пробелы
    text = RE_NEWLINES.sub('\n\n', text)            # >2 переносов → 2

    # Пробел перед пунктуацией / после открывающих скобок
    text = re.sub(r'\s+([.,!?;:)\]}»"…])', r'\1', text)
    text = re.sub(r'([(\[{«"])\s+', r'\1', text)

    text = text.strip()

    # Проверка длины
    if len(text) < min_len or len(text) > max_len:
        return None

    return text


# Дедупликация


# Синематика





