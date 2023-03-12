#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

import typing as t

import numpy as np
import torch
import torch.nn as nn

from feature_attributions.type import (
    InputTensorGeneric,
    TargetIndex
)

__all__ = ["GradCAM"]


class GradCAMExtractor(object):
    # Extract tensors needed for GradCAM using hooks
    _features: torch.Tensor
    _feature_gradient: torch.Tensor

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer

        self.layer.register_backward_hook(self._extract_layer_grads)
        self.layer.register_forward_hook(self._extract_layer_features)

    def get_features_and_gradients(self, x: torch.Tensor, target_idx: t.Optional[torch.Tensor]):
        out = self.model(x)

        if target_idx is None:
            target_idx = out.data.max(1, keepdim=True)[1]

        output_scalar = -nn.functional.nll_loss(out, target_idx.flatten(), reduction="sum")

        # Compute gradients
        self.model.zero_grad()
        output_scalar.backward()

        return self._features, self._feature_gradient

    # noinspection PyUnusedLocal
    def _extract_layer_grads(self, module: nn.Module, in_grad: t.Tuple[torch.Tensor], out_grad: t.Tuple[torch.Tensor]):
        # function to collect the gradient outputs
        self._feature_gradient = out_grad[0]

    # noinspection PyUnusedLocal
    def _extract_layer_features(self, module: nn.Module, inputs: t.Tuple[torch.Tensor], outputs: torch.Tensor):
        # function to collect the layer outputs
        self._features = outputs


class GradCAM(object):
    r"""GradCAM"""

    def __init__(self, model: nn.Module):
        self.model = model

        parameter = next(self.model.parameters())
        self._dtype = parameter.dtype
        self._device = parameter.device
        self._extractor = GradCAMExtractor(self.model, self._select_target_layer())

    def explain(
            self,
            x: InputTensorGeneric,
            target_idx: TargetIndex = None
    ) -> np.ndarray:
        if isinstance(x, (list, tuple)):
            if torch.is_tensor(x[0]):
                x = torch.concat(x, dim=0)
            else:
                x = np.concatenate(x, axis=0)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(dtype=self._dtype, device=self._device)

        if target_idx is None:
            target_idx = [None for _ in range(0, len(x))]
        elif isinstance(target_idx, int):
            index = target_idx
            target_idx = [index for _ in range(0, len(x))]

        assert len(x) == len(target_idx)

        attributions = list()
        for x_instance, target_index in zip(x, target_idx):
            if target_index is not None:
                target_index = torch.tensor(target_index).to(dtype=torch.long, device=self._device)
            attributions.append(
                self._compute_attribution(
                    x=x_instance.unsqueeze(0),
                    target_idx=target_index,
                ).detach().cpu().numpy()
            )
        attributions = np.concatenate(attributions, axis=0)

        return attributions

    def _compute_attribution(self, x: torch.Tensor, target_idx: t.Optional[torch.Tensor] = None):
        self.model.eval()
        features, intermed_grad = self._extractor.get_features_and_gradients(x, target_idx=target_idx)

        # GradCAM computation
        grads = intermed_grad.mean(dim=(2, 3), keepdim=True)
        cam = (nn.functional.relu(features) * grads).sum(1, keepdim=True)
        attribution = nn.functional.interpolate(
            nn.functional.relu(cam), size=x.size(2), mode="bilinear", align_corners=True
        )

        return attribution

    def _select_target_layer(self) -> nn.Module:
        # Iterate through layers
        prev_module = None
        target_module = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                prev_module = m
            elif isinstance(m, nn.Linear):
                target_module = prev_module
                break

        return target_module
