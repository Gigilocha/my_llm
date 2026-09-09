import torch

from src.data.sft_format import IGNORE_INDEX


# Паддинг батча SFT-примеров разной длины до максимальной длины В ЭТОМ батче
# (не до глобального max_len — экономия compute на батчах с короткими примерами).
# attention_mask: 1=реальный токен, 0=паддинг — уходит в Transformer.forward()
def collate_sft_batch(
    examples: list[tuple[list[int], list[int]]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len_in_batch = max(len(ids) for ids, _ in examples)

    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for input_ids, labels in examples:
        pad_len = max_len_in_batch - len(input_ids)
        input_ids_batch.append(input_ids + [pad_token_id] * pad_len)
        labels_batch.append(labels + [IGNORE_INDEX] * pad_len)
        attention_mask_batch.append([1] * len(input_ids) + [0] * pad_len)

    return (
        torch.tensor(input_ids_batch, dtype=torch.long),
        torch.tensor(labels_batch, dtype=torch.long),
        torch.tensor(attention_mask_batch, dtype=torch.long),
    )
