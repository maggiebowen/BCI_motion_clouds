"""
utils.py: Core utility functions for DynTex and MotionCloud.

Provides:
- diff1: first-order finite-difference convolution operator.
- diff2: second-order finite-difference convolution operator.
- periodic_comp: compute the periodic component of a spatial image via Fourier methods.
"""
import torch as tch
from torch import pi as PI

# first finite differences
# diff1 computes D1: f[n+1] - f[n]
diff1 = tch.nn.Conv1d(
    in_channels=1, out_channels=1,
    kernel_size=2, stride=1,
    padding=0, groups=1, bias=False
)
kernel = tch.tensor([[[-1.0, 1.0]]], dtype=tch.float32)
diff1.weight.data = kernel
diff1.weight.requires_grad = False

# second finite differences
# diff2 computes D2: f[n+1] - 2f[n] + f[n-1]
diff2 = tch.nn.Conv1d(
    in_channels=1, out_channels=1,
    kernel_size=3, stride=1,
    padding=0, groups=1, bias=False
)
kernel = tch.tensor([[[1.0, -2.0, 1.0]]], dtype=tch.float32)
diff2.weight.data = kernel
diff2.weight.requires_grad = False


def periodic_comp(im: tch.Tensor, freq: bool, dev: tch.device) -> tch.Tensor:
    """
    Compute the periodic component of a 2D image using Fourier decomposition.

    Parameters
    ----------
    im : Tensor
        Input image of shape (M, N).
    freq : bool
        If True, return the Fourier-domain component; if False, return the spatial-domain result.
    dev : torch.device
        Device for computation (e.g., 'cpu' or 'cuda').

    Returns
    -------
    Tensor
        If freq is True: complex-valued Fourier component of shape (M, N).
        If freq is False: real-valued spatial image of shape (M, N), inverse FFT of the periodic component.
    """
    # Convert to double precision for stability
    M, N = im.shape
    im = tch.tensor(im, dtype=tch.float64, device=dev)

    # Edge differences for periodic boundary handling
    top = im[-1, :] - im[0, :]
    bottom = im[0, :] - im[-1, :]
    left = im[:, -1] - im[:, 0]
    right = im[:, 0] - im[:, -1]
    v = tch.zeros_like(im)
    v[0, :] = top
    v[-1, :] = bottom
    v[:, 0] += left
    v[:, -1] += right

    # Discrete Laplacian with periodic boundaries
    lap = -4.0 * im
    lap[:, :-1] += im[:, 1:]
    lap[:, 1:]  += im[:, :-1]
    lap[:-1, :] += im[1:, :]
    lap[1:, :]  += im[:-1, :]
    lap[0, :]   += im[-1, :]
    lap[-1, :]  += im[0, :]
    lap[:, 0]   += im[:, -1]
    lap[:, -1]  += im[:, 0]

    # Fourier transform of (Laplacian - edge term)
    diff_im_v = tch.fft.fft2(lap - v, norm='ortho')

    # Build denominator array avoiding division by zero
    Lx = tch.arange(N, device=dev)
    Ly = tch.arange(M, device=dev)
    X, Y = tch.meshgrid(Lx, Ly, indexing='xy')
    div = 2.0*tch.cos(2*PI*X/N) + 2.0*tch.cos(2*PI*Y/M) - 4.0
    div[0, 0] = 1.0

    # Periodic component in Fourier domain
    per_fft = diff_im_v / div
    per_fft[0, 0] = tch.sum(im)

    if freq:
        return per_fft
    else:
        # Inverse FFT to spatial domain
        return tch.fft.ifftn(per_fft, norm='ortho').real
