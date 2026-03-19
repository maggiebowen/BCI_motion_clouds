"""
DriftingGrating: Generate a drifting sinusoidal grating stimulus using PyTorch tensors.

This module extends DynTex to produce a simple spatiotemporal sine grating defined by
spatial frequency, orientation, contrast, and drift velocity.
"""

import torch as tch
from math import pi as PI
from .DynTex import DynTex


class DriftingGrating(DynTex):
    """
    DriftingGrating generates a drifting sine-wave grating stimulus frame-by-frame.

    Attributes
    ----------
    N : int
        Spatial dimension (pixels) of the square grating.
    frame_per_second : int
        Frame rate (Hz) used to increment the temporal phase.
    pixel_per_cm : float
        Display density for conversion between cycles/degree and pixels.
    dt : float
        Time step per frame = 1/frame_per_second.
    contrast : float
        Contrast of the grating (0–1 range).
    sf : torch.Tensor
        Spatial frequency in cycles/pixel.
    th : torch.Tensor
        Orientation angle in degrees.
    v : torch.Tensor
        Drift velocity in cycles/frame.
    t : torch.Tensor
        Current temporal phase (frames since start).
    x, y : torch.Tensor
        Meshgrid coordinates in normalized [-0.5, +0.5] range.
    """

    def __init__(
        self,
        directory=None,
        filename=None,
        dev=None,
        N=512,
        frame_per_second=60,
        pixel_per_cm=65
    ):
        """
        Initialize the DriftingGrating object.

        Parameters
        ----------
        directory, filename : None
            Unused (inherited from DynTex).
        dev : torch.device or None
            Computation device; defaults to CPU or CUDA if available.
        N : int, optional
            Spatial size of the grating (pixels). Default is 512.
        frame_per_second : int, optional
            Frame rate for drift in Hz. Default is 60.
        pixel_per_cm : float, optional
            Display density for unit conversion. Default is 65.
        """
        super().__init__(directory, filename, dev)
        # basic parameters
        self.N = N
        self.frame_per_second = frame_per_second
        self.pixel_per_cm = pixel_per_cm
        self.dt = 1.0 / self.frame_per_second
        # placeholders for grating parameters
        self.contrast = None
        self.sf = None
        self.th = None
        self.v = None
        self.t = None
        self.x = None
        self.y = None

    def _dg(self, x, y, contrast, sf, th, v, t):
        """
        Internal: compute a single drifting grating frame.

        Parameters
        ----------
        x, y : Tensor
            Spatial coordinate grids.
        contrast : float
            Contrast of the grating (0–1).
        sf : Tensor
            Spatial frequency in cycles/pixel.
        th : Tensor
            Orientation in degrees.
        v : Tensor
            Drift velocity in cycles/frame.
        t : Tensor
            Temporal phase index (frames).

        Returns
        -------
        Tensor
            Grayscale image of shape (N, N) with values in [0, 255].
        """
        theta = th * PI / 180.0
        phase = v * t
        arg = 2 * PI * sf * (x * tch.cos(theta) + y * tch.sin(theta) + phase)
        sine_wave = tch.sin(arg)
        # scale to [0,255] with contrast and mean 127.5
        return contrast * 0.5 * sine_wave + 127.5

    def set_params(self, contrast, sf, th, v, t=0):
        """
        Set grating parameters and initialize spatial grid.

        Parameters
        ----------
        contrast : float
            Contrast of the grating (0–1).
        sf : float
            Spatial frequency in cycles/degree.
        th : float
            Orientation angle in degrees.
        v : float
            Drift speed in degrees/second.
        t : float, optional
            Initial temporal phase (frames). Default is 0.
        """
        # assign parameters
        self.contrast = contrast
        # convert cycles/degree to cycles/pixel
        self.sf = tch.tensor(self.N / self.pixel_per_cm * sf, device=self.dev)
        self.th = tch.tensor(th, device=self.dev)
        # convert degrees/sec to cycles/frame in pixel units
        self.v = tch.tensor(self.pixel_per_cm * v / (self.N * self.frame_per_second), device=self.dev)
        self.t = tch.tensor(t, device=self.dev)
        # create normalized meshgrid
        coords = tch.linspace(-0.5, 0.5, self.N, device=self.dev)
        self.x, self.y = tch.meshgrid(coords, coords, indexing='xy')

    def get_frame(self):
        """
        Generate and return the next drifting grating frame.

        Returns
        -------
        Tensor
            Grayscale image of shape (N, N) with values in [0, 255].

        Side effects
        -------------
        Increments the internal temporal phase t by 1 frame.
        """
        frame = self._dg(self.x, self.y, self.contrast, self.sf, self.th, self.v, self.t)
        # advance phase
        self.t = self.t + 1.0
        return frame
