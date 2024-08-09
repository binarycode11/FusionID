import numpy as np

def evaluate_matches(matches_matrix, threshold=0.5):
    """
    Avalia as correspondências entre conjuntos de referência e inspeção para determinar a precisão da identificação.

    Parâmetros:
    - matches_matrix (numpy.ndarray): Uma matriz de similaridade de dimensão n x m, onde n é o número de
      imagens na inspeção e m é o número de imagens na referência.
    - threshold (float): Limiar de similaridade para considerar uma identificação como válida.

    Retorna:
    - TP (int): Número de verdadeiros positivos.
    - FP (int): Número de falsos positivos.
    - FN (int): Número de falsos negativos.
    """
    n, m = matches_matrix.shape
    TP, FP, FN = 0, 0, 0

    # Calcular TP e FN
    for i in range(m):
        if np.max(matches_matrix[i]) >= threshold and np.argmax(matches_matrix[i]) == i:
            TP += 1
        else:
            FN += 1

    # Calcular FP
    for i in range(m, n):
        if np.max(matches_matrix[i]) >= threshold and np.argmax(matches_matrix[i]) < m:
            FP += 1

    return TP, FP, FN
