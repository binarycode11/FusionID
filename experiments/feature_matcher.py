import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from interfaces.pipeline import IGlobalMatcher

class FloydWarshall(IGlobalMatcher):
    @staticmethod
    def floydWarshall(graph):
        graph = csr_matrix(graph)
        dist_matrix, _ = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
        return dist_matrix

    @staticmethod
    def match_matrix(mat_a, mat_b, threshold):
        # Garantir que mat_a e mat_b não contenham valores inválidos
        mat_a = np.nan_to_num(mat_a)
        mat_b = np.nan_to_num(mat_b)
        mat_dist = mat_b - mat_a

        # Substituir valores inválidos por zero e limitar valores para evitar overflow
        mat_dist = np.nan_to_num(mat_dist)
        # Limitar os valores de mat_dist para evitar overflow
        mat_dist = np.clip(mat_dist, -1e150, 1e150)

        for i in range(mat_dist.shape[0]):
            mat_dist[i, :i] = 0
        mat_dist = mat_dist * mat_dist
        points = 0
        for i in range(mat_dist.shape[0]):
            for j in range(i + 1, mat_dist.shape[0]):
                if mat_dist[i, j] < threshold:
                    points += 1
        return points

    def __call__(self, matrixAdj0: np.ndarray, matrixAdj1: np.ndarray, threshold: float = 0.2) -> int:
        matAdjFull0 = self.floydWarshall(matrixAdj0)
        matAdjFull1 = self.floydWarshall(matrixAdj1)
        simGraph = self.match_matrix(matAdjFull0, matAdjFull1, threshold)
        return simGraph
