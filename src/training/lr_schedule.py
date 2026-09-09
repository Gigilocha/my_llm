import math


# Получение Lr в зависимости от момента обучения
def get_lr(step: int, learning_rate: float, min_learning_rate: float, warmup_steps: int, max_steps: int) -> float:
    
    if step < warmup_steps:
        lr = learning_rate * (step / warmup_steps)
    elif step < max_steps:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = min_learning_rate + (learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress)) / 2
    else:
        lr = min_learning_rate 

    return lr 
        

