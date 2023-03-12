import typing as t
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import visualization as viz
from matplotlib.figure import Figure
from matplotlib.pyplot import (
    axis,
    figure
)


class ImageVisualizationMethod(Enum):
    heat_map = 1
    blended_heat_map = 2
    original_image = 3
    masked_image = 4
    alpha_scaling = 5
    saliency = 6


def visualize_image_attr(
    attr: np.ndarray,
    original_image: t.Optional[np.ndarray] = None,
    method: str = "heatmap",
    sign: str = "absolute_value",
    plt_fig_axis: t.Optional[t.Tuple[figure, axis]] = None,
    outlier_perc: t.Union[int, float] = 2,
    cmap: t.Optional[str] = None,
    alpha_overlay: float = 0.5,
    show_colorbar: bool = False,
    title: t.Optional[str] = None,
    fig_size: t.Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
):
    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(figsize=fig_size)
        else:
            plt_fig = Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots()

    if original_image is not None:
        if np.max(original_image) <= 1.0:
            original_image = viz._prepare_image(original_image * 255)
    elif ImageVisualizationMethod[method] != ImageVisualizationMethod.heat_map:
        raise ValueError(
            "Original Image must be provided for"
            "any visualization other than heatmap."
        )

    # Remove ticks and tick labels from plot.
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_yticklabels([])
    plt_axis.set_xticklabels([])

    heat_map = None
    # Show original image
    if ImageVisualizationMethod[method] == ImageVisualizationMethod.original_image:
        assert (
                original_image is not None
        ), "Original image expected for original_image method."
        if len(original_image.shape) > 2 and original_image.shape[2] == 1:
            original_image = np.squeeze(original_image, axis=2)
        plt_axis.imshow(original_image)
    else:
        # Choose appropriate signed attributions and normalize.
        norm_attr = viz._normalize_attr(attr, sign, outlier_perc, reduction_axis=2)

        # Set default colormap and bounds based on sign.
        if viz.VisualizeSign[sign] == viz.VisualizeSign.all:
            default_cmap = viz.LinearSegmentedColormap.from_list(
                "RdWhGn", ["red", "white", "green"]
            )
            vmin, vmax = -1, 1
        elif viz.VisualizeSign[sign] == viz.VisualizeSign.positive:
            default_cmap = "Greens"
            vmin, vmax = 0, 1
        elif viz.VisualizeSign[sign] == viz.VisualizeSign.negative:
            default_cmap = "Reds"
            vmin, vmax = 0, 1
        elif viz.VisualizeSign[sign] == viz.VisualizeSign.absolute_value:
            default_cmap = "Blues"
            vmin, vmax = 0, 1
        else:
            raise AssertionError("Visualize Sign type is not valid.")
        cmap = cmap if cmap is not None else default_cmap

        # Show appropriate image visualization.
        if ImageVisualizationMethod[method] == ImageVisualizationMethod.heat_map:
            heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
        elif (
                ImageVisualizationMethod[method]
                == ImageVisualizationMethod.blended_heat_map
        ):
            assert (
                    original_image is not None
            ), "Original Image expected for blended_heat_map method."
            plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")
            heat_map = plt_axis.imshow(
                norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay
            )
        elif ImageVisualizationMethod[method] == ImageVisualizationMethod.masked_image:
            assert viz.VisualizeSign[sign] != viz.VisualizeSign.all, (
                "Cannot display masked image with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                viz._prepare_image(original_image * np.expand_dims(norm_attr, 2))
            )
        elif ImageVisualizationMethod[method] == ImageVisualizationMethod.alpha_scaling:
            assert viz.VisualizeSign[sign] != viz.VisualizeSign.all, (
                "Cannot display alpha scaling with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                np.concatenate(
                    [
                        original_image,
                        viz._prepare_image(np.expand_dims(norm_attr, 2) * 255),
                    ],
                    axis=2,
                )
            )
        elif ImageVisualizationMethod[method] == ImageVisualizationMethod.saliency:
            norm_attr = norm_attr - norm_attr.min()
            norm_attr = norm_attr / norm_attr.max()
            norm_attr = norm_attr.clip(0, 1)
            norm_attr = np.uint8(norm_attr * 255)

            color_heatmap = cv2.applyColorMap(norm_attr, cv2.COLORMAP_JET)  # noqa

            # Combine image with heatmap
            img_with_heatmap = np.float32(color_heatmap) + np.float32(original_image)
            img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
            img_with_heatmap = np.uint8(255 * img_with_heatmap)

            plt_axis.imshow(np.flip(img_with_heatmap, axis=2))
        else:
            raise AssertionError("Visualize Method type is not valid.")

    # Add colorbar. If given method is not a heatmap and no colormap is relevant,
    # then a colormap axis is created and hidden. This is necessary for appropriate
    # alignment when visualizing multiple plots, some with heatmaps and some
    # without.
    if show_colorbar:
        axis_separator = viz.make_axes_locatable(plt_axis)
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
        if heat_map:
            plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)
        else:
            colorbar_axis.axis("off")
    if title:
        plt_axis.set_title(title)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis


def visualize_image_attr_multiple(
    attr: np.ndarray,
    original_image: t.Optional[np.ndarray],
    methods: t.List[str],
    signs: t.List[str],
    titles: t.Optional[t.List[str]] = None,
    fig_size: t.Tuple[int, int] = (8, 6),
    use_pyplot: bool = True,
    **kwargs: t.Any,
):
    assert len(methods) == len(signs), "Methods and signs array lengths must match."
    if titles is not None:
        assert len(methods) == len(titles), (
            "If titles list is given, length must " "match that of methods list."
        )
    if use_pyplot:
        plt_fig = plt.figure(figsize=fig_size)
    else:
        plt_fig = Figure(figsize=fig_size)
    plt_axis = plt_fig.subplots(1, len(methods))

    # When visualizing one
    if len(methods) == 1:
        plt_axis = [plt_axis]

    for i in range(len(methods)):
        visualize_image_attr(
            attr,
            original_image=original_image,
            method=methods[i],
            sign=signs[i],
            plt_fig_axis=(plt_fig, plt_axis[i]),
            use_pyplot=False,
            title=titles[i] if titles else None,
            **kwargs,
        )

    plt_fig.tight_layout()

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis
