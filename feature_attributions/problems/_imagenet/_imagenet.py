import json
import os
import typing as t
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from einops import rearrange
from torchvision import transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

__all__ = ["ImageNetProblem"]


class ImageNetProblem(object):

    def __init__(self):
        self._dataset_path = Path(__file__).parents[0]
        with open(os.path.join(self._dataset_path, "imagenet1000_clsidx_to_labels.json")) as json_data:
            self._idx_to_labels = json.load(json_data)

    @property
    def image(self) -> np.ndarray:
        image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])(Image.open(os.path.join(self._dataset_path, "n01820546_14917.JPEG"))).unsqueeze(0)
        image = rearrange(image, "b c h w -> b h w c").detach().numpy()

        return image

    @property
    def normalizer(self) -> transforms.Normalize:
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def get_resnet18(self) -> t.Tuple[t.Tuple[np.ndarray, int], nn.Module]:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.eval()

        parameter = next(model.parameters())

        normalized_image = torch.from_numpy(self.image).to(dtype=parameter.dtype, device=parameter.device)
        normalized_image = rearrange(normalized_image, "b h w c -> b c h w")
        normalized_image = self.normalizer(normalized_image)

        output = model(normalized_image)
        output = nn.functional.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()
        predicted_label = self._idx_to_labels[str(pred_label_idx.item())][1]
        print("Predicted:", predicted_label, "(", prediction_score.squeeze().item(), ")", flush=True)

        normalized_image = rearrange(normalized_image, "b c h w -> b h w c").detach().cpu().numpy()

        return (normalized_image, pred_label_idx.item()), model


if __name__ == "__main__":
    ImageNetProblem().get_resnet18()
