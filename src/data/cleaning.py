import re
import unicodedata
from typing import Literal


# Ссылки — вырезаются только в mode="prose". В коде URL часто часть строки,
# комментария или docstring, а не мусор — вырезать его нельзя
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

# Управляющие символы (кроме \n и \t, их обрабатываем отдельно)
RE_CONTROL = re.compile(
    r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F'
    r'\u200B-\u200F\u2028-\u202F\u2060-\u206F\uFEFF\u00AD]'
)


"""
clean_text() работает в двух режимах — mode="prose" и mode="code" — а не
через набор независимых bool-флагов. Причина: с отдельными флагами
(strip_urls, collapse_punct, collapse_whitespace, ...) легко забыть выключить
что-то одно и незаметно сломать код-корпус (например, выключить collapse_punct,
но забыть collapse_whitespace — отступы Python всё равно разъедутся). Два
явных именованных режима исключают такую наполовину безопасную конфигурацию.

mode="prose" — полная чистка веб-текста: снятие ссылок, схлопывание повторной
пунктуации, удаление пустых скобок/кавычек, схлопывание пробелов и переносов.
Это артефакты HTML-скрейпинга (fineweb-2, wikipedia) — безопасно для обычного
текста, но КАТАСТРОФИЧНО для кода: схлопывание пробелов ломает синтаксически
значимые отступы Python, удаление "пустых скобок" вырезает легитимный код
(f(), пустой кортеж ()), схлопывание повторов портит разделители вида
"# ====" или docstring-баннеры.

mode="code" — только безопасная гигиена: Unicode NFC, управляющие символы,
фильтр по длине. Внутренние пробелы/переносы/пунктуация не трогаются вообще.
"""

CleanMode = Literal["prose", "code"]


# Если предложение/документ получается пустым или выходит за границы длины —
# возвращаем None (сигнал "выбросить документ"), иначе — очищенный текст
def clean_text(text: str, min_len: int, max_len: int, mode: CleanMode = "prose", max_punct: int = 3) -> str | None:
    if not isinstance(text, str):
        return None

    # Unicode-нормализация и управляющие символы — безопасны и нужны ВСЕГДА,
    # независимо от режима (это гигиена кодировки, не трансформация контента)
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text)

    if mode == "prose":
        # Удаление ссылок
        text = RE_URL.sub(" ", text)

        # Удаление пустых кавычек
        prev = None
        while prev != text:
            prev = text
            text = RE_EMPTY_QUOTES.sub("", text)
            text = re.sub(r'["\']\s*["\']', '', text)

        # Удаление пустых скобок
        prev_text = None
        while prev_text != text:
            prev_text = text
            text = RE_EMPTY_BRACKETS.sub("", text)
            text = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', text)

        # Повторяющаяся пунктуация
        text = RE_PUNCT_REPEAT.sub(lambda m: m.group(1) * max_punct, text)

        # Пробелы и переносы
        text = RE_SPACE_NL.sub('\n', text)               # пробелы вокруг \n
        text = RE_WS.sub(' ', text)                       # множественные пробелы
        text = RE_NEWLINES.sub('\n\n', text)               # >2 переносов → 2

        # Пробел перед пунктуацией / после открывающих скобок
        text = re.sub(r'\s+([.,!?;:)\]}»"…])', r'\1', text)
        text = re.sub(r'([(\[{«"])\s+', r'\1', text)

        text = text.strip()
    else:  # mode == "code"
        # Только внешние пустые строки — внутренние отступы/пробелы/пустые
        # строки трогать нельзя, они синтаксически значимы
        text = text.strip("\n")

    if len(text) < min_len or len(text) > max_len:
        return None

    return text


# Лёгкая чистка для уже курированных SFT/DPO-примеров — только Unicode-
# нормализация и управляющие символы, без фильтра по длине (это решает
# _extract_sft_example/_extract_dpo_example проверкой на непустоту) и без
# URL/пунктуационной чистки: инструкционные диалоги могут легитимно содержать
# ссылки (если сам вопрос про них) или код — тот же риск поломки, что у
# mode="code" в clean_text()
def light_clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return RE_CONTROL.sub("", text).strip()


# Дедупликация (точные/почти-дубликаты — MinHash/эмбеддинги) и семантическая
# фильтрация (оценка качества текста моделью) — намеренно НЕ реализованы в
# этом проходе. Это отдельный кусок инфраструктуры (нужен собственный индекс
# для дедупа, отдельная модель-скорер для семантики), не то же самое по
# объёму, что "почистить артефакты скрейпинга" — решаем отдельно, после
# первого полного прогона, не блокируя эту неделю