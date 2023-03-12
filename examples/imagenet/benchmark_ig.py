import typing as t

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients as CaptumIG
from matplotlib.colors import LinearSegmentedColormap
from saliency.core import IntegratedGradients as SaliencyIG

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


def benchmark_captum(**kwargs):
    ig = CaptumIG(forward_func=kwargs["model"])
    attribution = ig.attribute(
        inputs=torch.from_numpy(np.transpose(kwargs["image"].copy(), (0, 3, 1, 2))).to(dtype=torch.float32),
        baselines=torch.from_numpy(np.transpose(kwargs["baselines"].copy(), (0, 3, 1, 2))).to(dtype=torch.float32),
        target=kwargs["target_idx"]
    ).detach().cpu().numpy()
    attribution = np.transpose(attribution, (0, 2, 3, 1))

    fig = plot_image_attribution(kwargs["image"][0, ...], attribution[0, ...])
    fig.savefig("benchmark_ig_captum.png")


def benchmark_saliency(**kwargs):
    call_model_args = {
        "model": kwargs["model"],
        "target_idx": kwargs["target_idx"]
    }

    ig = SaliencyIG()
    attribution = ig.GetMask(
        x_value=kwargs["image"][0, ...].copy(),
        call_model_function=call_model_function,
        call_model_args=call_model_args,
        x_baseline=kwargs["baselines"][0, ...].copy(),
        x_steps=kwargs["n_steps"]
    )

    fig = plot_image_attribution(kwargs["image"][0, ...], attribution)
    fig.savefig("benchmark_ig_saliency.png")


def main():
    problem = ImageNetProblem()
    image = problem.image
    normalizer = problem.normalizer
    (normalized_image, target_idx), model = problem.get_resnet18()
    wrapped_model = ModelWrapper(model, normalizer).to(dtype=torch.float32)
    baselines = np.zeros_like(image)

    params = {
        "model": wrapped_model,
        "image": np.asarray(image, dtype=np.float32),
        "baselines": np.asarray(baselines, dtype=np.float32),
        "target_idx": target_idx,
        "n_steps": 25
    }

    benchmark_captum(**params)
    benchmark_saliency(**params)


if __name__ == "__main__":
    main()
