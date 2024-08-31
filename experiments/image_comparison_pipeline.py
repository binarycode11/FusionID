import numpy as np
import torch
from kornia.feature import LocalFeature, DescriptorMatcher
from kornia.utils import tensor_to_image
from typing import Dict, Tuple, List
from utils import MyDrawMatcher, evaluate_matches, print_table  # Supondo que essas funções estão em 'utils.py'
from experiments import DelaunayGraph
class ImageComparisonPipeline:
    def __init__(self, preprocessor=None, local_feature: LocalFeature = None, descriptor_matcher: DescriptorMatcher = None):
        """
        Inicializa o pipeline de comparação de imagens.

        Args:
            preprocessor (optional): Pré-processador de imagens.
            local_feature (LocalFeature, optional): Extrator de características locais.
            descriptor_matcher (DescriptorMatcher, optional): Comparador de descritores.

        """
        self.preprocessor = preprocessor
        self.local_feature = local_feature
        self.descriptor_matcher = descriptor_matcher

    def run(self, inspection_images: torch.Tensor, reference_images: torch.Tensor, threshold=0.1, log=None, device=torch.device('cpu')) -> Dict[Tuple[int, int], float]:
        """
        Executa o pipeline de comparação de imagens.

        Args:
            inspection_images (torch.Tensor): Imagens de inspeção.
            reference_images (torch.Tensor): Imagens de referência.
            threshold (float): Limiar para a comparação global.
            log (str, optional): Nível de log para controle de saída.
            device (torch.device, optional): Dispositivo para execução.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuplas contendo contagens e escores de correspondência.
        """
        if not all([self.preprocessor, self.local_feature, self.descriptor_matcher]):
            raise ValueError("Pipeline components are not fully set.")

        n, m = inspection_images.shape[0], reference_images.shape[0]
        scores = np.zeros((n, m))
        count_match = np.zeros((n, m))
        myDraw = MyDrawMatcher()
        cache_reference = {}

        for i_index, i_image in enumerate(inspection_images):
            lafs0, responses0, descriptors0 = self.local_feature(i_image[:1][None])
            for r_index, r_image in enumerate(reference_images):
                if r_index not in cache_reference:
                    lafs1, responses1, descriptors1 = self.local_feature(r_image[:1][None])
                    cache_reference[r_index] = (lafs1, responses1, descriptors1)
                else:
                    lafs1, responses1, descriptors1 = cache_reference[r_index]

                distance, matches = self.descriptor_matcher(descriptors0[0], descriptors1[0])
                out = {
                    "keypoints0": lafs0[0, :, :, 2].data,
                    "keypoints1": lafs1[0, :, :, 2].data,
                    "lafs0": lafs0,
                    "lafs1": lafs1,
                    "descriptors0": descriptors0[0],
                    "descriptors1": descriptors1[0],
                    "matches": matches,
                }

                num_match = matches.shape[0]
                if num_match < 4:
                    num_match = 0

                count_match[i_index, r_index] = num_match


        if log is not None and log in ('INFO'):
            print_table(count_match)
            print("count_match : ", evaluate_matches(count_match, 8))

        return count_match
