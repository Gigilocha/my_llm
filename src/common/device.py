import os
import torch
import torch.distributed as dist


"""
Определение устройства и задел под распределённое обучение.

Одна GPU (текущий сценарий): resolve_device() решает, что использовать
("auto" -> cuda, если доступна, иначе cpu; либо явно заданное значение с проверкой).

Несколько GPU (когда понадобится): torchrun сам расставляет переменные окружения
(RANK, LOCAL_RANK, WORLD_SIZE) при запуске `torchrun --nproc_per_node=N script.py`.
setup_distributed()/cleanup_distributed() читают их и поднимают/гасят process group.
Без torchrun этих переменных нет — is_distributed() вернёт False, всё работает
как раньше на одной GPU/CPU, ничего не ломается для текущего сценария.
"""


# Определение устройства.
# requested="auto" -> cuda при наличии, иначе cpu.
# requested="cuda"/"cpu" -> используется как есть, но если попросили cuda,
# а её нет — не падаем молча в cpu, а громко предупреждаем (тихий fallback
# на cpu на 126M модели даст не ошибку, а "почему-то очень медленно")
def resolve_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("⚠️  Запрошен device=cuda, но CUDA недоступна. Использую cpu.")
        return "cpu"

    return requested


# Человекочитаемая информация об устройстве — для логов при старте обучения
def get_device_info(device: str) -> str:
    if device == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        capability = torch.cuda.get_device_capability(idx)
        return f"cuda:{idx} ({name}, {total_mem_gb:.1f} GB, compute capability {capability[0]}.{capability[1]})"

    return "cpu"


# --- Distributed (задел на будущее, активируется только под torchrun) ---

# Запущено ли обучение под torchrun (RANK/WORLD_SIZE в окружении)
def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


# Инициализация process group. backend="nccl" для GPU (быстрый, но только CUDA),
# "gloo" — универсальный fallback для CPU-кластеров/отладки без GPU.
# Возвращает device для текущего процесса — на multi-GPU это НЕ просто "cuda",
# а "cuda:{local_rank}", иначе все процессы на ноде полезут в одну и ту же GPU
def setup_distributed(backend: str | None = None) -> str:
    if not is_distributed():
        raise RuntimeError("setup_distributed() вызван без torchrun (нет RANK/WORLD_SIZE в окружении)")

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)

    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"

    return "cpu"


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
