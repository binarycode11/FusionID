import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import kornia as K  # Importe kornia se não já estiver importado

class MyDrawMatcher:
    def __init__(self, draw_dict=None) -> None:
        if draw_dict is None:
            draw_dict = {
                "inlier_color": (0.2, 1, 0.2),
                "tentative_color": (1.0, 0.5, 1),
                "feature_color": (0.6, 0.5, 0,0),
                "vertical": True,
            }
        self.draw_dict = draw_dict

    def __call__(self, img1_preprocessed, img2_preprocessed, output) -> None:
        from kornia_moons.viz import draw_LAF_matches
        # Use kornia.tensor_to_image to ensure the images are in the correct format for drawing
        img1 = K.tensor_to_image(img1_preprocessed.squeeze())
        img2 = K.tensor_to_image(img2_preprocessed.squeeze())

        # Assumes draw_LAF_matches is accessible and compatible with provided arguments
        draw_LAF_matches(
            output['lafs0'].cpu(),
            output['lafs1'].cpu(),
            output['matches'].cpu(),
            img1,
            img2,
            None,  # Or the inliers if available
            self.draw_dict
        )
        # Remove as bordas numéricas no plot (se aplicável)
        plt.tick_params(labelbottom=False, labelleft=False)  # Desativa os labels nos eixos
        plt.show()  # Exibe o plot


def plot_tensor(tensor):
    """
    Plota um tensor PyTorch como uma imagem.

    Parâmetros:
    - tensor (torch.Tensor): Um tensor 2D para imagens em escala de cinza ou um tensor 3D para imagens RGB.
    """
    if tensor.dim() == 2:
        plt.imshow(tensor, cmap='gray')
    elif tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
        plt.imshow(tensor)
    else:
        raise ValueError("O tensor deve ser 2D (imagem em escala de cinza) ou 3D (imagem RGB).")
    plt.axis('off')
    plt.show()

def plot_image_with_keypoints(image, keypoints):
    """
    Plota uma imagem e seus keypoints.

    Parâmetros:
    - image (torch.Tensor): A imagem original.
    - keypoints (torch.Tensor): Os keypoints detectados.
    """
    if image.dim() == 3:
        image = image.permute(1, 2, 0)
    if torch.max(image) > 1:
        image = image / 255.0
    plt.imshow(image.cpu().numpy())
    if keypoints.dim() == 3:
        keypoints = keypoints[0]
    if keypoints.shape[1] == 2:
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=20, marker='.', c='r')
    elif keypoints.shape[1] == 3:
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=20 * keypoints[:, 2], marker='.', c='r')
    plt.axis('off')
    plt.show()
