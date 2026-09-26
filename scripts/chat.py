import argparse

from transformers import PreTrainedTokenizerFast

from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.data.format import format_prompt_for_generation
from src.model.build import build_model
from src.training.checkpoint import find_latest_checkpoint, load_checkpoint
from src.engine.select import make_generate_fn


"""
Интерактивный чат в терминале — ручная проверка модели глазами, не
автоматическая оценка (для неё есть base_eval.py/sft_eval.py). Сохраняет
историю диалога (multi-turn), пока не наберёшь /reset или /exit.
"""


def main():
    parser = argparse.ArgumentParser(description="Интерактивный чат с моделью в терминале")
    parser.add_argument("--checkpoint", choices=["sft", "pretrain"], default="sft",
                         help="Какие веса использовать (по умолчанию — SFT)")
    parser.add_argument("--step", type=int, default=None, help="Номер шага (по умолчанию — последний доступный)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    args = parser.parse_args()

    config = get_config()
    device = resolve_device(config.env.device)
    print(f"Устройство: {get_device_info(device)}")

    engine_cfg = config.engine.model_copy(update={
        k: v for k, v in {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }.items() if v is not None
    })

    checkpoints_dir_name = "checkpoints_sft" if args.checkpoint == "sft" else "checkpoints"
    checkpoints_dir = PROJECT_ROOT / "outputs" / checkpoints_dir_name
    step = args.step if args.step is not None else find_latest_checkpoint(checkpoints_dir)
    if step is None:
        raise FileNotFoundError(
            f"Нет чекпоинтов в {checkpoints_dir}. "
            f"{'Запусти sft_train.py' if args.checkpoint == 'sft' else 'Запусти base_train.py'} сначала, "
            f"либо укажи --checkpoint pretrain, если SFT ещё не готов."
        )

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    special_tokens = config.tokenizer.special_tokens

    model = build_model(config).to(device)
    # device=device — грузим чекпоинт напрямую на нужное устройство, важно
    # именно здесь: chat.py — типичный сценарий "быстро проверить модель на
    # ноутбуке без GPU", а чекпоинт почти наверняка сохранён с CUDA-машины
    load_checkpoint(checkpoint_dir=checkpoints_dir, step=step, model=model, device=device)
    model.eval()

    generate_fn = make_generate_fn(model, tokenizer, device, config.model.model.max_position_embeddings, engine_cfg)

    print(f"Чекпоинт: {args.checkpoint}, шаг {step}. Движок: "
          f"{'GenerationEngine/KV-кеш' if engine_cfg.use_kv_cache else 'naive'}.")
    print("Команды: /reset — очистить историю, /exit — выйти.\n")

    messages: list[dict] = []

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/reset":
            messages = []
            print("(история очищена)\n")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt_text = format_prompt_for_generation(messages, special_tokens)

        try:
            response = generate_fn(prompt_text, return_full_text=False).strip()
        except ValueError as e:
            # Переполнение max_position_embeddings — не должно убивать всю сессию,
            # просто откатываем последний ход и просим сбросить историю
            print(f"(Ошибка: {e}\nПопробуй /reset — история диалога стала слишком длинной)\n")
            messages.pop()
            continue

        # На pretrain-чекпоинте (без SFT) спецтокены/формат диалога модель не
        # обучена завершать корректно — ответ может не остановиться на EOS
        # естественным образом, это ожидаемо и не баг чата
        print(f"Модель: {response}\n")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()