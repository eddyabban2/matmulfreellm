# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch


def onnx_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """ONNX/TRT-friendly HGRN recurrence without a Python loop over sequence length.

    The legacy ``naive_recurrent_hgrn`` uses ``for i in range(T)``, which
    torch.onnx.export unrolls at the trace length (e.g. T=8). Runtime Gather
    nodes then fail for any other sequence length.

    Uses the closed form h[t] = P[t] * (h0 + cumsum(x / P)) with
    P = exp(cumsum(log(g))) (ONNX opset 17 has CumSum but not CumProd).
    """
    dtype = x.dtype
    xf, gf = x.float(), g.float()
    # cumprod is not in ONNX opset 17; exp(cumsum(log(g))) is equivalent for g > 0.
    log_g = torch.log(gf.clamp(min=1e-20))
    P = torch.exp(torch.cumsum(log_g, dim=2))
    eps = 1e-20
    x_scaled = xf / P.clamp(min=eps)
    h_cumsum = torch.cumsum(x_scaled, dim=2)
    if initial_state is not None:
        o = P * (initial_state.unsqueeze(2).float() + h_cumsum)
    else:
        o = P * h_cumsum
    final_state = o[:, :, -1, :] if output_final_state else None
    return o.to(dtype), final_state


def naive_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    dtype = x.dtype
    x, g = map(lambda i: i.float(), (x, g))
    B, H, T, D = x.shape

    h = torch.zeros(B, H, D, dtype=torch.float, device=x.device)
    o = torch.zeros_like(x)

    final_state = None
    if initial_state is not None:
        h += initial_state.detach()

    for i in range(T):
        h = g[:, :, i] * h + x[:, :, i]
        o[:, :, i] = h

    if output_final_state:
        final_state = h
    return o.to(dtype), final_state
