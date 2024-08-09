import torch
import kornia as K
from torch import nn
from interfaces.pipeline import IPreprocessor

class PreprocessPipeline(IPreprocessor):
    def __init__(self):
        super(PreprocessPipeline, self).__init__()
        self.transforms = nn.Sequential(
            K.geometry.Resize((200, 200)),
            K.color.RgbToGrayscale()
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        x = self.transforms(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x
