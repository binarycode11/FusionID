import gc
import torch
import numpy as np
import time
from contextlib import contextmanager

def free_memory():
    """
    Libera a memória ocupada pelo garbage collector e o cache da GPU, se disponível.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_seed(seed):
    """
    Define a semente para reprodutibilidade em PyTorch, CUDA e NumPy.

    Parâmetros:
    - seed (int): O valor da semente.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@contextmanager
def medir_tempo(label: str = "Tempo de execução"):
    """
    Medidor de tempo de execução para blocos de código.

    Parâmetros:
    - label (str): Rótulo a ser exibido com o tempo de execução.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label}: {end - start} segundos")
