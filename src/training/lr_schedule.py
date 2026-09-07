import math

# Получение LR в зависимости от текущего шага обучения
def get_lr(
    step: int, 
    learning_rate: float, 
    min_learning_rate: float, 
    warmup_steps: int, 
    max_steps: int
) -> float:
    
    # 1. Линейный Warmup
    if step < warmup_steps:
        if warmup_steps == 0:
            return learning_rate
        return learning_rate * (step / warmup_steps)
    
    # 2. Cosine Decay
    if step < max_steps:
        decay_steps = max_steps - warmup_steps
        if decay_steps <= 0:
            return min_learning_rate
        
        progress = (step - warmup_steps) / decay_steps
        cos_decay = (1.0 + math.cos(math.pi * progress)) / 2.0
        return min_learning_rate + (learning_rate - min_learning_rate) * cos_decay
    
    # 3. После достижения max_steps
    return min_learning_rate