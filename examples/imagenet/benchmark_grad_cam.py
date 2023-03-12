import typing as t

import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap

from feature_attributions.algorithms import GradCAM
from feature_attributions.problems import ImageNetProblem
from feature_attributions.utils.visualization import visualize_image_attr_multiple


class ModelWrapper(nn.Module):

    def __init__(self, model: nn.Module, preprocess: t.Optional[t.Callable[[torch.Tensor], torch.Tensor]]):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.preprocess = preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self.preprocess is not None:
            x = self.preprocess(x)
        return self.model(x)


# noinspection PyUnusedLocal
def call_model_function(images, call_model_args=None, expected_keys=None):
    model = call_model_args["model"]
    target_idx = call_model_args["target_idx"]

    parameter = next(model.parameters())

    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.from_numpy(images).to(dtype=parameter.dtype, device=parameter.device)
    images = images.requires_grad_(True)

    output = model(images)
    output = nn.functional.softmax(output, dim=1)

    outputs = output[:, target_idx]
    gradients = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
    gradients = torch.movedim(gradients[0], 1, 3)
    gradients = gradients.detach().cpu().numpy()

    return {"INPUT_OUTPUT_GRADIENTS": gradients}


def plot_image_attribution(image: np.ndarray, attribution: np.ndarray):
    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )
    fig, _ = visualize_image_attr_multiple(
        attribution,
        image,
        ["original_image", "blended_heat_map", "saliency"],
        ["all", "positive", "positive"],
        fig_size=(12, 6),
        cmap=default_cmap,
        show_colorbar=True
    )

    return fig


def benchmark_feature_attributions(**kwargs):
    guided_ig = GradCAM(model=kwargs["model"])
    attribution = guided_ig.explain(
        x=np.transpose(kwargs["image"].copy(), (0, 3, 1, 2)),
        target_idx=kwargs["target_idx"]
    )
    attribution = np.transpose(attribution, (0, 2, 3, 1))

    fig = plot_image_attribution(kwargs["image"][0, ...], attribution[0, ...])
    fig.savefig("benchmark_grad_cam_feature_attributions.png")


def main():
    problem = ImageNetProblem()
    image = problem.image
    normalizer = problem.normalizer
    (normalized_image, target_idx), model = problem.get_resnet18()
    wrapped_model = ModelWrapper(model, normalizer).to(dtype=torch.float64)

    params = {
        "model": wrapped_model,
        "image": np.asarray(image, dtype=np.float64),
        "target_idx": target_idx
    }

    benchmark_feature_attributions(**params)


if __name__ == "__main__":
    main()
