import typing as t

import numpy as np
import torch

InputTensorGeneric = t.TypeVar(
    "InputTensorGeneric",
    np.ndarray,
    torch.Tensor,
    t.List[t.Union[np.ndarray, torch.Tensor]],
    t.Tuple[t.Union[np.ndarray, torch.Tensor], ...]
)
Baseline = t.Optional[
    t.Union[
        int,
        float,
        np.ndarray,
        torch.Tensor,
        t.List[t.Union[int, float, np.ndarray, torch.Tensor]],
        t.Tuple[t.Union[int, float, np.ndarray, torch.Tensor], ...]
    ]
]
TargetIndex = t.Optional[
    t.Union[
        int,
        t.List[int],
        t.Tuple[int, ...],
        np.ndarray,
        torch.Tensor
    ]
]
