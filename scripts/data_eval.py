import argparse
import statistics

from transformers import PreTrainedTokenizerFast

from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.dataset import build_language_mix_from_disk, extract_texts
from src.tokenizer.tokenizer import encode


"""
Проверки данных, которые нужно прогонять ПОСЛЕ кеширования и ПЕРЕД обучением —
чтобы находить проблемы вроде "20% кода в конфиге — это на деле 3% токенов"
или "val пересекается с train" до того, как на это будет потрачено 50000 шагов
компьюта, а не после.

1. Калибровка пропорций микса: *_quantity в data_config.yaml — это веса выбора
   ДОКУМЕНТА, а не итоговая доля ТОКЕНОВ. Если средняя длина документа сильно
   разнится между языками/доменами (у нас код в разы короче текстовых статей),
   даже "честные" 60/20/20 дают совсем другую пропорцию токенов на выходе.
2. Достаточность val: после фикса утечки (train и val больше не пересекаются)
   актуальный вопрос — хватает ли val данных, чтобы оценка была на РАЗНЫХ
   документах при усреднении по многим батчам, а не гоняла один и тот же
   маленький кусок по кругу (что снова снижает дисперсию оценки не за счёт
   реальной статистики, а за счёт повторного использования тех же примеров).
3. Пересечение train/val: прямая проверка на дубликаты текста между train и val
   как защитная сетка на будущее — если кто-то (в т.ч. я сам) снова сломает
   логику кеширования, это всплывёт здесь, а не через подозрительно низкий
   val_loss через 50000 шагов.
"""


def measure_avg_length(sources, stage, language, data_dir, seed, tokenizer, split, sample_size) -> tuple[float, list[str]]:
    mix = build_language_mix_from_disk(sources, stage, language, data_dir, seed, split=split)
    texts = []
    lengths = []
    for text in extract_texts(mix):
        texts.append(text)
        lengths.append(len(encode(tokenizer, text, add_special_tokens=False)))
        if len(lengths) >= sample_size:
            break
    return statistics.mean(lengths) if lengths else 0.0, texts


def check_mix_calibration(config, data_dir, tokenizer, sample_size, logger) -> None:
    cfg = config.data.pre_training_data
    target_tokens = {"rus": cfg.rus_quantity, "en": cfg.en_quantity, "code": cfg.code_quantity}
    sources = {"rus": cfg.rus_sources, "en": cfg.en_sources, "code": cfg.code_sources}

    logger.info(f"=== 1. Калибровка пропорций микса (n={sample_size} на язык, train) ===")
    avg_lengths = {}
    for language in ("rus", "en", "code"):
        avg_len, _ = measure_avg_length(sources[language], "pretrain", language, data_dir, cfg.seed, tokenizer, "train", sample_size)
        avg_lengths[language] = avg_len
        logger.info(f"  {language}: средняя длина документа = {avg_len:.0f} токенов")

    total_target = sum(target_tokens.values())
    total_current_tokens_weighted = sum(target_tokens[l] / total_target * avg_lengths[l] for l in avg_lengths)

    logger.info("  Текущие веса (доля документов) vs фактическая доля токенов:")
    for language in ("rus", "en", "code"):
        doc_share = target_tokens[language] / total_target * 100
        token_share = (target_tokens[language] / total_target * avg_lengths[language]) / total_current_tokens_weighted * 100
        drift = abs(token_share - doc_share)
        flag = "  ⚠️  большое расхождение" if drift > 10 else ""
        logger.info(f"    {language}: doc_weight={doc_share:.1f}%  ->  token_share≈{token_share:.1f}%{flag}")

    raw_weights = {l: (target_tokens[l] / total_target) / avg_lengths[l] for l in avg_lengths}
    total_raw = sum(raw_weights.values())
    corrected = {l: raw_weights[l] / total_raw * 100 for l in avg_lengths}

    logger.info(f"  Рекомендуемые *_quantity для целевой доли токенов {target_tokens}:")
    for language in ("rus", "en", "code"):
        logger.info(f"    {language}_quantity: {corrected[language]:.2f}")


def check_val_adequacy(config, data_dir, tokenizer, sample_size, logger) -> None:
    cfg = config.data.pre_training_data
    pretrain_cfg = config.training.pre_training
    sources = {"rus": cfg.rus_sources, "en": cfg.en_sources, "code": cfg.code_sources}
    cache_docs = {"rus": cfg.rus_cache_docs, "en": cfg.en_cache_docs, "code": cfg.code_cache_docs}

    # Сколько токенов val расходуется за ОДИН вызов base_eval.py при типичном
    # --num-eval-batches (берём консервативно 50, как дефолт в base_eval.py)
    tokens_per_eval_call = 50 * pretrain_cfg.batch_size * pretrain_cfg.max_len

    logger.info(f"\n=== 2. Достаточность val (текущий val_split_ratio={cfg.val_split_ratio}) ===")
    for language in ("rus", "en", "code"):
        avg_len, _ = measure_avg_length(sources[language], "pretrain", language, data_dir, cfg.seed, tokenizer, "val", sample_size)
        val_docs_estimate = int(cache_docs[language] * cfg.val_split_ratio)
        total_val_tokens = avg_len * val_docs_estimate

        ratio = total_val_tokens / tokens_per_eval_call if tokens_per_eval_call else float("inf")
        if ratio < 1:
            verdict = "⚠️  val МЕНЬШЕ одного прохода оценки — часть данных переиспользуется в пределах одного eval"
        elif ratio < 3:
            verdict = "⚠️  val заметно переиспользуется (мало независимых документов относительно объёма оценки)"
        else:
            verdict = "OK — достаточно данных для оценки без сильного переиспользования"
        logger.info(f"  {language}: ≈{total_val_tokens:,.0f} токенов в val (≈{val_docs_estimate} докум.), "
                    f"нужно ≈{tokens_per_eval_call:,} на один eval-прогон — {verdict}")


def check_train_val_overlap(config, data_dir, tokenizer, sample_size, logger) -> None:
    cfg = config.data.pre_training_data
    sources = {"rus": cfg.rus_sources, "en": cfg.en_sources, "code": cfg.code_sources}

    logger.info(f"\n=== 3. Проверка пересечения train/val (n={sample_size} на сторону) ===")
    any_overlap = False
    for language in ("rus", "en", "code"):
        _, train_texts = measure_avg_length(sources[language], "pretrain", language, data_dir, cfg.seed, tokenizer, "train", sample_size)
        _, val_texts = measure_avg_length(sources[language], "pretrain", language, data_dir, cfg.seed, tokenizer, "val", sample_size)
        overlap = set(train_texts) & set(val_texts)
        if overlap:
            any_overlap = True
            logger.info(f"  {language}: ⚠️  {len(overlap)} совпадающих документов из {sample_size} — УТЕЧКА ДАННЫХ")
        else:
            logger.info(f"  {language}: пересечений не найдено (выборка {sample_size} на сторону)")

    if any_overlap:
        logger.info("  ⚠️  Обнаружена утечка train/val — не запускай обучение, пока не разберёшься с кешированием")
    else:
        logger.info("  Выборочная проверка чистая (не гарантия на 100% всех данных, но хороший сигнал)")


def show_sample_data(config, data_dir, tokenizer, logger):
    """Показывает пример документа из каждого датасета."""
    cfg = config.data.pre_training_data
    sources = {"rus": cfg.rus_sources, "en": cfg.en_sources, "code": cfg.code_sources}
    
    logger.info("\n=== 0. Примеры данных из каждого источника ===")
    for language, source_list in sources.items():
        for source in source_list:
            try:
                mix = build_language_mix_from_disk(
                    [source], "pretrain", language, data_dir, cfg.seed, split="train"
                )
                for i, text in enumerate(extract_texts(mix)):
                    if i >= 1:  # только первый документ
                        break
                    logger.info(f"  [{language}] {source.dataset_name}:")
                    logger.info(f"    {text[:200]}...")  # первые 200 символов
            except Exception as e:
                logger.warning(f"  [{language}] {source.dataset_name}: не удалось загрузить пример — {e}")

def main():
    parser = argparse.ArgumentParser(description="Проверка данных перед обучением: калибровка микса, val, утечки")
    parser.add_argument("--sample-size", type=int, default=300, help="Сколько документов на язык мерить в каждой проверке")
    args = parser.parse_args()

    config = get_config()
    logger = setup_logger(__name__)
    data_dir = PROJECT_ROOT / config.env.data_dir

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    show_sample_data(config, data_dir, tokenizer, logger)

    check_mix_calibration(config, data_dir, tokenizer, args.sample_size, logger)
    check_val_adequacy(config, data_dir, tokenizer, args.sample_size, logger)
    check_train_val_overlap(config, data_dir, tokenizer, args.sample_size, logger)


if __name__ == "__main__":
    main()
