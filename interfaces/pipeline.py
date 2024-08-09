from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np
import torch

class IPreprocessor(ABC):
    """
    Interface para um pré-processador de imagens.
    Define o contrato para classes que implementam operações de pré-processamento em imagens.
    """
    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Aplica operações de pré-processamento à imagem fornecida.

        Parâmetros:
            image (torch.Tensor): A imagem de entrada como um tensor do PyTorch.

        Retorna:
            torch.Tensor: A imagem após o pré-processamento.
        """
        pass

class IGlobalFeatureStructurer(ABC):
    """
    Interface para a estruturação global de features através de grafo.
    """
    @abstractmethod
    def __call__(self, points: np.ndarray, featuresByPoints: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Estrutura features globais de uma imagem em um grafo, utilizando pontos e suas características associadas.

        Args:
            points (np.ndarray): Array de pontos extraídos de uma imagem, onde cada ponto é uma coordenada 2D.
            featuresByPoints (np.ndarray): Array de características associadas a cada ponto.

        Returns:
            Tuple[np.ndarray, Any]: Um grafo representando a estruturação global das features e o objeto Delaunay utilizado para a triangulação.
        """
        pass

class IGlobalMatcher(ABC):
    """
    Interface para a similaridade global de features através de grafo.
    """
    @abstractmethod
    def __call__(self, matrixAdj0: np.ndarray, matrixAdj1: np.ndarray, threshold: float = 0.2) -> int:
        """
        Interface para invocar operações de comparação de grafos.

        Args:
            matrixAdj0: A primeira matriz de adjacência do grafo.
            matrixAdj1: A segunda matriz de adjacência do grafo.
            threshold: O limiar para avaliar correspondências.

        Returns:
            Um valor representando a similaridade entre os dois grafos.
        """
        pass
