import pytest
from src.data.cleaning import clean_text


# --- Базовые ссылки, скобки, кавычки ---

def test_removes_url():
    text = "Посмотрите видео https://example.com/watch?v=123 прямо сейчас"
    result = clean_text(text, min_len=5, max_len=1000)
    assert "example.com" not in result
    assert "Посмотрите видео" in result


def test_removes_empty_parentheses_after_url_stripped():
    # пустые скобки, оставшиеся после удаления ссылки
    text = "Инструкция доступна на сайте ()."
    result = clean_text(text, min_len=5, max_len=1000)
    assert "()" not in result


def test_removes_nested_empty_parentheses():
    text = "Текст (( )) с вложенными пустыми скобками."
    result = clean_text(text, min_len=5, max_len=1000)
    assert "(" not in result and ")" not in result


def test_removes_empty_quotes_various_styles():
    text = 'Проблема "" возникает у новичков «» часто.'
    result = clean_text(text, min_len=5, max_len=1000)
    assert '""' not in result
    assert "«»" not in result


def test_keeps_non_empty_quotes_untouched():
    # непустые кавычки — не мусор, трогать нельзя
    text = 'Сайт называется «самоучитель для чайников».'
    result = clean_text(text, min_len=5, max_len=1000)
    assert "«самоучитель для чайников»" in result


# --- Повторы пунктуации ---

def test_caps_repeated_punctuation_at_max_three():
    text = "Это очень важно!!!!!!!"
    result = clean_text(text, min_len=5, max_len=1000)
    assert result.endswith("!!!")
    assert "!!!!" not in result


def test_does_not_touch_single_punctuation():
    text = "Всё хорошо! Продолжаем..."
    result = clean_text(text, min_len=5, max_len=1000)
    assert "хорошо!" in result
    assert "продолжаем..." in result.lower() or "Продолжаем..." in result


# --- Пробелы и переносы ---

def test_collapses_multiple_spaces():
    text = "Слово    с лишними     пробелами."
    result = clean_text(text, min_len=5, max_len=1000)
    assert "  " not in result


def test_keeps_paragraph_break_but_collapses_extra_newlines():
    text = "Первый абзац.\n\n\n\nВторой абзац."
    result = clean_text(text, min_len=5, max_len=1000)
    assert "\n\n\n" not in result
    assert "\n\n" in result  # граница абзаца сохранена


def test_removes_space_before_punctuation():
    text = "Привет , как дела ?"
    result = clean_text(text, min_len=5, max_len=1000)
    assert " ," not in result
    assert " ?" not in result


# --- Защитные тесты: не мусор ---

def test_keeps_cyrillic_and_diacritics_untouched():
    text = "Кофе café и ёлка не должны потеряться."
    result = clean_text(text, min_len=5, max_len=1000)
    assert "café" in result
    assert "ёлка" in result


def test_keeps_large_numbers_untouched():
    text = "Модель обучалась на 1000000 примеров текста."
    result = clean_text(text, min_len=5, max_len=1000)
    assert "1000000" in result


def test_keeps_short_legit_repeated_sentences():
    # дедуп предложений не входит в v1 — короткие повторы не трогаем
    text = "Да. Да. Именно так."
    result = clean_text(text, min_len=5, max_len=1000)
    assert result.count("Да.") == 2


# --- Длина ---

def test_returns_none_for_text_shorter_than_min_len():
    text = "Коротко."
    result = clean_text(text, min_len=50, max_len=1000)
    assert result is None


def test_returns_none_for_text_longer_than_max_len():
    text = "Слово " * 100
    result = clean_text(text, min_len=5, max_len=20)
    assert result is None


def test_returns_valid_text_within_length_bounds():
    text = "Это нормальный текст подходящей длины для прохождения проверки."
    result = clean_text(text, min_len=10, max_len=200)
    assert result is not None
    assert isinstance(result, str)


# --- Не строка на входе ---

def test_returns_none_for_non_string_input():
    assert clean_text(None, min_len=5, max_len=1000) is None
    assert clean_text(123, min_len=5, max_len=1000) is None