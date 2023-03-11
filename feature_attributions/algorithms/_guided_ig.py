import numpy as np
import torch
import torch.nn as nn

from feature_attributions.type import (
    Baseline,
    InputTensorGeneric,
    TargetIndex
)

__all__ = ["GuidedIntegratedGradients"]


class GuidedIntegratedGradients(object):

    def __init__(self, model: nn.Module):
        self.model = model

        parameter = next(self.model.parameters())
        self._dtype = parameter.dtype
        self._device = parameter.device

    def explain(
            self,
            x: InputTensorGeneric,
            baselines: Baseline = None,
            target_idx: TargetIndex = None,
            n_steps: int = 200,
            fraction: float = 0.25,
            max_l1_distance: float = 0.02
    ) -> np.ndarray:
        if isinstance(x, (list, tuple)):
            if torch.is_tensor(x[0]):
                x = torch.concat(x, dim=0)
            else:
                x = np.concatenate(x, axis=0)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(dtype=self._dtype, device=self._device)

        if baselines is None:
            baselines = torch.zeros_like(x, dtype=self._dtype, device=self._device)
        elif isinstance(baselines, (int, float)):
            baselines = torch.ones_like(x, dtype=self._dtype, device=self._device) * baselines
        elif isinstance(baselines, np.ndarray):
            baselines = torch.from_numpy(baselines).to(dtype=self._dtype, device=self._device)
        elif isinstance(baselines, (list, tuple)):
            baselines_ = list()
            for baseline in baselines:
                if isinstance(baselines, (int, float)):
                    baseline = torch.ones(x.shape[1:], dtype=self._dtype, device=self._device) * baseline
                    baseline = baseline.unsqueeze(0)
                elif isinstance(baseline, np.ndarray):
                    baseline = torch.from_numpy(baseline).to(dtype=self._dtype, device=self._device)
                else:
                    baseline = baseline.to(dtype=self._dtype, device=self._device)
                baselines_.append(baseline)
            baselines = torch.concat(baselines_, dim=0)

        if target_idx is None:
            target_idx = [None for _ in range(0, len(x))]
        elif isinstance(target_idx, int):
            index = target_idx
            target_idx = [index for _ in range(0, len(x))]

        assert len(x) == len(baselines) == len(target_idx)

        attributions = list()
        for x_instance, baseline, target_index in zip(x, baselines, target_idx):
            attributions.append(
                self._compute_attribution(
                    x_input=x_instance,
                    baseline=baseline,
                    target_idx=target_index,
                    n_steps=n_steps,
                    fraction=fraction,
                    max_l1_distance=max_l1_distance
                ).detach().cpu().unsqueeze(0).numpy()
            )
        attributions = np.concatenate(attributions, axis=0)

        return attributions

    def _compute_attribution(
            self,
            x_input: torch.Tensor,
            baseline: torch.Tensor,
            target_idx: int,
            n_steps: int = 200,
            fraction: float = 0.25,
            max_l1_distance: float = 0.02
    ) -> torch.Tensor:
        """Calculates and returns Guided IG attribution.

        Args:
            x_input: model input that should be explained.
            baseline: chosen baseline for the input explanation.
            n_steps: the number of Riemann sum steps for path integral approximation.
            fraction: the fraction of features [0, 1] that should be selected and
                changed at every approximation step. E.g., value `0.25` means that 25% of
                the input features with the smallest gradients are selected and changed at
                every step.
            max_l1_distance: the relative maximum L1 distance [0, 1] that any feature can
                deviate from the straight line path. Value `0` allows no deviation and,
                therefore, corresponds to the Integrated Gradients method that is
                calculated on the straight-line path. Value `1` corresponds to the
                unbounded Guided IG method, where the path can go through any point within
                the baseline-input hyper-rectangular.
        """
        attribution = torch.zeros_like(x_input).to(dtype=self._dtype, device=self._device)

        # If the input is equal to the baseline then the attribution is zero.
        if torch.abs(x_input - baseline).sum() == 0:
            return attribution

        x_point = baseline.clone().detach()
        total_l1_distance = torch.abs(x_input - baseline).sum()

        # Iterate through every step.
        for step in range(n_steps):
            # Calculate gradients and make a copy.
            feature_gradients_actual = self._get_feature_gradients(x_point, target_idx)
            feature_gradients = feature_gradients_actual.clone().detach()
            # Calculate current step alpha and the ranges of allowed values for this step.
            alpha = (step + 1.0) / n_steps
            alpha_min = max(alpha - max_l1_distance, 0.0)
            alpha_max = min(alpha + max_l1_distance, 1.0)
            x_min = self._translate_alpha_to_point(alpha_min, x_input, baseline)
            x_max = self._translate_alpha_to_point(alpha_max, x_input, baseline)
            # The goal of every step is to reduce L1 distance to the input.
            # `target_l1_distance` is the desired L1 distance after completion of this step.
            target_l1_distance = total_l1_distance * (1 - (step + 1) / n_steps)

            # Iterate until the desired L1 distance has been reached.
            gamma = np.inf
            while gamma > 1.0:
                previous_x_point = x_point.clone().detach()
                x_alpha = self._translate_point_to_alpha(x_point, x_input, baseline)
                x_alpha[torch.isnan(x_alpha)] = alpha_max
                # All features that fell behind the [alpha_min, alpha_max] interval in terms of alpha,
                # should be assigned the x_min values.
                x_point[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

                # Calculate current L1 distance from the input.
                current_l1_distance = torch.abs(x_point - x_input).sum()
                # If the current L1 distance is close enough to the desired one then
                # update the attribution and proceed to the next step.
                if torch.isclose(target_l1_distance, current_l1_distance, rtol=1e-9, atol=1e-9):
                    attribution += (x_point - previous_x_point) * feature_gradients_actual
                    break

                # Features that reached `x_max` should not be included in the selection.
                # Assign very high gradients to them, so they are excluded.
                feature_gradients[x_point == x_max] = np.inf

                # Select features with the lowest absolute gradient.
                threshold = torch.quantile(torch.abs(feature_gradients), fraction, interpolation="lower")
                selected_feature = torch.logical_and(
                    torch.abs(feature_gradients) <= threshold, feature_gradients != np.inf  # noqa
                )

                # Find by how much the L1 distance can be reduced by changing only the selected features.
                selected_feature_l1_distance = (torch.abs(x_point - x_max) * selected_feature).sum().item()

                # Calculate ratio `gamma` that show how much the selected features should
                # be changed toward `x_max` to close the gap between current L1 and target L1.
                if selected_feature_l1_distance > 0:
                    gamma = (current_l1_distance - target_l1_distance) / selected_feature_l1_distance
                else:
                    gamma = np.inf

                if gamma > 1.0:
                    # Gamma higher than 1.0 means that changing selected features is not enough to close the gap.
                    # Therefore, change them as much as possible to stay in the valid range.
                    x_point[selected_feature] = x_max[selected_feature]
                else:
                    assert gamma > 0, gamma
                    x_point[selected_feature] = self._translate_alpha_to_point(gamma, x_max, x_point)[selected_feature]

                # Update attribution to reflect changes in `x_point`.
                attribution += (x_point - previous_x_point) * feature_gradients_actual

        return attribution

    def _get_feature_gradients(self, x_value: torch.Tensor, target_idx: int) -> torch.Tensor:
        images = x_value.unsqueeze(0)
        images = images.requires_grad_(True)

        output = self.model(images)
        output = nn.functional.softmax(output, dim=1)

        outputs = output[:, target_idx]
        gradients = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))[0]

        return gradients.squeeze(0)

    @staticmethod
    def _translate_point_to_alpha(x_point: torch.Tensor, x_input: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        """Translates a point on straight-line path to its corresponding alpha value.

        Args:
            x_point: the point on the straight-line path.
            x_input: the end point of the straight-line path.
            baseline: the start point of the straight-line path.

        Returns:
            The alpha value in range [0, 1] that shows the relative location of the
            point between x_baseline and x_input.
        """
        with torch.no_grad():
            return torch.where(x_input - baseline != 0, (x_point - baseline) / (x_input - baseline), np.nan)

    @staticmethod
    def _translate_alpha_to_point(alpha: float, x_input: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        """Translates alpha to the point coordinates within straight-line interval.

        Args:
            alpha: the relative location of the point between x_baseline and x_input.
            x_input: the end point of the straight-line path.
            baseline: the start point of the straight-line path.

        Returns:
            The coordinates of the point within [x_baseline, x_input] interval
            that correspond to the given value of alpha.
        """
        assert 0 <= alpha <= 1.0
        return baseline + (x_input - baseline) * alpha
