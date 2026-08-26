import pytest


# Очистка кеша
@pytest.fixture(autouse=True)
def _clean_config_cache():
    from src.common.config import get_env_settings, get_config
    get_env_settings.cache_clear()
    get_config.cache_clear()
    yield


# Ограничение на использование реального енв файла
@pytest.fixture(autouse=True)
def _isolate_from_real_env_file(monkeypatch, tmp_path):
    # почему: реальный .env в корне проекта не должен утекать в тесты —
    # без этого EnvSettings может найти значения из настоящего .env,
    # даже если тест явно их не задавал через monkeypatch.setenv
    monkeypatch.chdir(tmp_path)
    yield


import logging

# Закрытие handlers после каждого теста
@pytest.fixture(autouse=True)
def _close_all_loggers():
    yield
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)