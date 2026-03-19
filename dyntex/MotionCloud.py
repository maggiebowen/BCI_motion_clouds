"""
MotionCloud: Generate and process dynamic textures (MotionCloud stimuli) using PyTorch.

This module extends DynTex to produce spatiotemporal noise stimuli with defined
spectral kernels, AR(2) recursion, and optional phase translation.
"""

import torch as tch
import numpy as np
from math import pi as PI
from .DynTex import DynTex
from .utils import periodic_comp, diff1, diff2


class MotionCloud(DynTex):
    """
    MotionCloud generates MotionCloud stimuli by extending DynTex functionality.

    Attributes
    ----------
    over_samp : int
        Temporal oversampling factor.
    offset_duration : float
        Burn-in time (seconds) before sampling starts.
    N : int
        Spatial size (pixels) of the square stimulus.
    frame_per_second : int
        Display frame rate (Hz).
    pixel_per_cm : float
        Display density for unit conversions.
    ave_lum : float
        Mean luminance (0–255).
    contrast : float
        Contrast scaling factor.
    dt : float
        Time step per sub-frame = 1/(over_samp * frame_per_second).
    verbose : int
        Verbosity level for diagnostics.
    frame0, frame1, frame2 : torch.Tensor
        Buffers for AR(2) recursion in Fourier domain.
    spatial_frame : torch.Tensor
        Last output frame in spatial domain.
    noise_frame : torch.Tensor
        Last noise-only frame in spatial domain.
    fourier_translate, fourier_translate_start : torch.Tensor
        Phase terms for Fourier-domain translation.
    hist_len : int
        Length of circular buffers for past frames.
    frame_hist, composite_frame_hist : torch.Tensor
        Buffers storing previous FFT frames.
    frame_count, frame_count_speed : int
        Counters for frames generated and phase resets.
    x, y, R, Theta : torch.Tensor
        Spatial grid coordinates and polar transforms.
    th, th_sig : float
        Orientation preference (rad) and bandwidth.
    sf_mode, tp, octave : float, float, int
        Spatial frequency mode, temporal period, and octave flag.
    speed_vec : list[float]
        [vx, vy] motion vector in cycles/frame.
    rho, rho_sig, speed_sig : float
        Radial frequency and speed bandwidth parameters.
    al, be : torch.Tensor
        AR(2) recursion coefficients.
    spatial_kernel : torch.Tensor
        Precomputed spectral envelope.
    speed_pxpf : list[float]
        Translation speed in pixels/frame.
    """

    def __init__(
        self,
        directory=None,
        filename=None,
        dev=None,
        over_samp=5,
        offset_duration=2,
        N=512,
        frame_per_second=50,
        pixel_per_cm=65,
        ave_lum=127.0,
        contrast=35.0,
        verbose=0,
    ):
        """
        Initialize MotionCloud with specified parameters.

        Parameters
        ----------
        directory, filename : None
            Unused (inherited from DynTex).
        dev : torch.device or None
            Computation device; defaults to CUDA if available.
        over_samp : int
            Temporal oversampling factor.
        offset_duration : float
            Burn-in duration in seconds.
        N : int
            Spatial dimension in pixels.
        frame_per_second : int
            Frame rate in Hz.
        pixel_per_cm : float
            Display density.
        ave_lum : float
            Mean luminance.
        contrast : float
            Contrast scaling.
        verbose : int
            Print diagnostic messages if >0.
        """
        super().__init__(directory, filename, dev)
        # assign
        self.over_samp = over_samp
        self.offset_duration = offset_duration
        self.N = N
        self.frame_per_second = frame_per_second
        self.pixel_per_cm = pixel_per_cm
        self.ave_lum = ave_lum
        self.contrast = contrast
        self.verbose = verbose
        # compute
        self.dt = 1.0 / (self.over_samp * self.frame_per_second)
        # initialize buffers and attributes
        dev = self.dev
        self.frame0 = tch.zeros((N, N), device=dev)
        self.frame1 = tch.zeros((N, N), device=dev)
        self.frame2 = tch.zeros((N, N), device=dev)
        self.spatial_frame = tch.zeros((N, N), device=dev)
        self.noise_frame = tch.zeros((N, N), device=dev)
        self.fourier_translate = tch.ones((N, N), dtype=tch.cfloat, device=dev)
        self.fourier_translate_start = tch.ones((N, N), dtype=tch.cfloat, device=dev)
        self.hist_len = 128
        self.frame_hist = tch.zeros((self.hist_len, N, N), dtype=tch.cfloat, device=dev)
        self.composite_frame_hist = tch.zeros((self.hist_len, N, N), dtype=tch.cfloat, device=dev)
        self.frame_count = 0
        self.frame_count_speed = 0
        # placeholders
        self.x = self.y = self.R = self.Theta = None
        self.th = self.th_sig = self.sf_mode = self.tp = self.octave = None
        self.speed_vec = None
        self.rho = self.rho_sig = self.speed_sig = None
        self.al = self.be = self.spatial_kernel = None
        self.speed_pxpf = None

    def set_grids(self):
        """
        Create normalized spatial grids and polar coordinate arrays.

        Sets:
        - x, y : meshgrid in [-0.5, +0.5)
        - R     : radial distance
        - Theta : angle coordinate
        """
        lx_sup = tch.linspace(-self.N//2, -1, self.N//2, device=self.dev)
        lx_inf = tch.linspace(0, self.N//2-1, self.N//2, device=self.dev)
        Lx = tch.cat((lx_inf, lx_sup))
        x, y = tch.meshgrid(Lx, Lx, indexing='xy')
        self.x = x / self.N
        self.y = y / self.N
        R = tch.sqrt(self.x**2 + self.y**2)
        R[0, 0] = 1e-6
        self.R = R
        self.Theta = tch.atan2(self.y, self.x)

    def set_parameters(self, sf, sf_sig, th, th_sig, tf, spd_scalar, spd_dir, octa):
        """
        Compute spatiotemporal parameters for the spectral kernel.

        Parameters
        ----------
        sf, sf_sig : float
            Spatial frequency and its bandwidth (c/°).
        th, th_sig : float
            Orientation and its bandwidth (degrees).
        tf        : float
            Temporal frequency (Hz).
        spd_scalar, spd_dir : float
            Speed magnitude and direction (degrees).
        octa : int
            Octave flag (1 for bandwidth in c/°, 0 for octave calculation).
        """
        self.th = th * PI / 180
        self.th_sig = th_sig * PI / 180
        self.sf_mode = sf / self.pixel_per_cm
        self.tp = 1.0 / tf
        self.octave = octa
        angle = spd_dir * PI / 180
        self.speed_vec = [spd_scalar * np.cos(angle), spd_scalar * np.sin(angle)]
        # bandwidth u depends on octave
        if octa == 1:
            self.sf_sig = sf_sig
            u = np.sqrt(np.exp((sf_sig/np.sqrt(8)*np.sqrt(np.log(2)))**2) - 1)
        else:
            self.sf_sig = sf_sig / self.pixel_per_cm
            roots = np.roots([1,0,3,0,3,0,1,0,-(self.sf_sig/self.sf_mode)**2])
            u = np.real(roots[np.isreal(roots) & (np.real(roots)>0)][0])
        self.rho = tch.tensor(self.sf_mode * (1.0 + u**2))
        self.rho_sig = tch.tensor(u)
        self.speed_sig = 1/(self.rho*self.tp)

        safety_const = 1.0
        if self.verbose:
            if self.speed_sig<safety_const*(4-2*np.sqrt(2))/(self.N*self.dt):
                print('tf=%f must be lower than %f \n'
                      %(1/self.tp,(((4-2*np.sqrt(2))*self.rho)/(self.dt))))
            else:
                print('Correct parameters tf = %f < %f \n'
                      % (1/self.tp,(((-2*np.sqrt(2)+4)*self.rho)/(self.dt))))
        elif self.speed_sig<safety_const*(4-2*np.sqrt(2))/(self.N*self.dt):
                print('tf=%f must be lower than %f \n'
                      %(1/self.tp,(((4-2*np.sqrt(2))*self.rho)/(self.dt))))


    def set_ar_coeffs(self):
        """
        Compute AR(2) recursion coefficients from bandwidth and grid R.

        Sets:
        - al, be : complex AR coefficients for frame recursion.
        """
        one_over_tau = self.speed_sig * self.R
        a = 2 * one_over_tau
        b = one_over_tau**2
        self.al = 2 - self.dt*a - self.dt**2*b
        self.be = -1 + self.dt*a

    def set_fourier_translation(self):
        """
        Update Fourier-domain translation phases based on speed vector.

        Updates:
        - fourier_translate_start
        - fourier_translate
        - resets frame_count_speed
        """
        multiplier = self.pixel_per_cm/(self.frame_per_second*self.over_samp)
        self.speed_pxpf = [v*multiplier for v in self.speed_vec]
        self.fourier_translate_start *= self.fourier_translate**(self.frame_count_speed*self.over_samp)
        self.frame_count_speed = 0
        self.fourier_translate = tch.exp(PI*1j*(self.speed_pxpf[0]*self.x + self.speed_pxpf[1]*self.y))

    def bandpass_kernel(self):
        """
        Build a bandpass spectral envelope based on orientation and radial bandwidth.

        Sets:
        - spatial_kernel : spectral magnitude envelope tensor.
        """
        one_over_tau = self.speed_sig*self.R
        angular = tch.exp(tch.cos(2*(self.Theta-self.th))/(4*self.th_sig**2))
        radial = tch.exp(-(tch.log(self.R/self.rho)**2/ tch.log(1+self.rho_sig**2))/2)*(1.0/self.R)
        kernel = angular * radial*(1.0/self.R)**2
        pre_kernel = 4*(one_over_tau*self.dt)**3
        kernel *= pre_kernel
        C0 = 1.0/tch.sum(kernel)
        C1 = 1.0/tch.sum(pre_kernel)
        self.spatial_kernel = tch.sqrt(C0*kernel)
        self.pre_spatial_kernel = tch.sqrt(C1*pre_kernel)

    def nat_kernel(self):
        """
        Build a naturalistic spatial envelope with a simple low-pass and radial falloff.

        Sets:
        - spatial_kernel : spectral envelope tensor.
        """
        one_over_tau = self.speed_sig*self.R
        radial = (self.R < 0.7)
        kernel = radial*(1.0/self.R)**2*4*(one_over_tau*self.dt)**3
        C = 1.0/tch.sum(kernel)
        self.spatial_kernel = self.contrast*1.1e6/(2*PI)*tch.sqrt(C*kernel)/self.N

    def learn_kernel(self, f_mov, dt):
        """
        Estimate AR coefficients and spatial kernel from sample data.

        Parameters
        ----------
        f_mov : Tensor
            Input video tensor of shape (ny, nx, nf).
        dt : float
            Time step between frames.
        """
        ny, nx, nf = f_mov.shape
        D2 = diff2(f_mov)/dt**2
        D1 = diff1(f_mov)/dt
        # moment matrices and cross-terms
        M1 = tch.sum(tch.abs(D1)**2, axis=-1)
        M2 = tch.sum(tch.abs(f_mov)**2, axis=-1)
        Md = tch.sum(tch.conj(D1)*f_mov[..., :-1], axis=-1)
        # second-order terms
        N1 = -tch.sum(D2*tch.conj(D1[..., :-1]), axis=-1)
        N2 = -tch.sum(D2*tch.conj(f_mov[..., :-1]), axis=-1)
        # solve per-pixel
        a = tch.zeros((ny,nx), dtype=tch.cfloat, device=self.dev)
        b = tch.zeros((ny,nx), dtype=tch.cfloat, device=self.dev)
        for i in range(ny):
            for j in range(nx):
                if (i,j)!=(0,0):
                    A = tch.stack([tch.stack([M1[i,j], Md[i,j]]),
                                   tch.stack([tch.conj(Md[i,j]), M2[i,j]])])
                    B = tch.stack([N1[i,j], N2[i,j]])
                    if tch.linalg.det(A) < 1e-8:
                        x = tch.zeros(2, device=self.dev)
                    else:
                        x = tch.linalg.solve(A, B)
                    a[i,j], b[i,j] = x[0], x[1]
        self.a = tch.conj(a)
        self.b = tch.conj(b)
        self.al = 2 - self.dt*a - self.dt**2*b
        self.be = -1 + self.dt*a

    def set_all(self, kernel_type, sf, sf_sig, th, th_sig, tf, spd_scalar, spd_dir, octa):
        """
        Configure all parameters, grids, AR coeffs, and select kernel.

        Parameters
        ----------
        kernel_type : str
            'bandpass' or 'natural'.
        sf, sf_sig, th, th_sig, tf, spd_scalar, spd_dir, octa : parameters
            Passed to set_parameters and set_kernel.
        """
        self.set_parameters(sf, sf_sig, th, th_sig, tf, spd_scalar, spd_dir, octa)
        self.set_grids()
        self.set_ar_coeffs()
        self.set_kernel(kernel_type)

    def set_kernel(self, kernel_type):
        """
        Choose and build the specified spatial kernel.

        Parameters
        ----------
        kernel_type : str
            'bandpass' or 'natural'.
        """
        if kernel_type == "bandpass":
            self.bandpass_kernel()
        elif kernel_type == "natural":
            self.nat_kernel()

    def burnout(self, offset_duration=None):
        """
        Burn-in sequence by generating and discarding frames.

        Parameters
        ----------
        offset_duration : float, optional
            Override burn-in time in seconds.
        """
        if offset_duration is not None:
            self.offset_duration = offset_duration
        n_frames = int(self.offset_duration // self.dt)
        for _ in range(n_frames):
            self.get_frame()

    def _get_noise_frame(self):
        """
        Generate a noise spectrum frame in Fourier domain.

        Returns
        -------
        f_noise_frame : Tensor
            Complex noise spectrum weighted by spatial kernel.
        """
        noise = tch.randn((self.N, self.N), device=self.dev) + 1j*tch.randn((self.N, self.N), device=self.dev)
        return self.pre_spatial_kernel * noise

    def get_frame(self, adjust=False, store_noise_frame=False):
        """
        Generate the next spatial-domain frame.

        Parameters
        ----------
        adjust : bool, optional
            If True, normalize by std dev and apply contrast/ave_lum.
        store_noise_frame : bool, optional
            If True, save the noise-only frame.

        Returns
        -------
        spatial_frame : Tensor
            Real-valued frame after IFFT and optional adjustment.
        """
        # AR recursion for oversampled steps
        for _ in range(self.over_samp):
            f_noise = self._get_noise_frame()
            self.frame0 = self.al*self.frame1 + self.be*self.frame2 + f_noise
            self.frame2, self.frame1 = self.frame1, self.frame0

        # apply translation and IFFT
        phase = self.fourier_translate_start * self.fourier_translate**(self.frame_count_speed*self.over_samp)
        self.frame = phase * self.spatial_kernel * self.frame0
        self.spatial_frame = tch.fft.ifftn(self.frame, norm='forward').real

        if store_noise_frame:
            self.noise_frame = tch.fft.ifftn(f_noise, norm='forward').real

        # optional rescale
        if adjust:
            self.spatial_frame = (self.spatial_frame / self.spatial_frame.std())
            self.spatial_frame = self.contrast*self.spatial_frame + self.ave_lum
            if store_noise_frame:
                self.noise_frame = (self.noise_frame / self.noise_frame.std())
                self.noise_frame = self.contrast*self.noise_frame + self.ave_lum

        # update history
        idx = self.frame_count % self.hist_len
        self.frame_hist[idx] = self.frame
        self.frame_count += 1
        self.frame_count_speed += 1

        return self.spatial_frame

    def update_composite_frame_hist(self, compositeFrame):
        """
        Append the composite frame to its circular history buffer.

        Parameters
        ----------
        compositeFrame : Tensor
            Frame to store of shape (N, N).
        """
        idx = self.frame_count % self.hist_len
        self.composite_frame_hist[idx] = compositeFrame

    def syn_tex(self, nf, adjust=False):
        """
        Synthesize an RGB video by repeating get_frame outputs.

        Parameters
        ----------
        nf : int
            Number of frames to generate.
        adjust : bool, optional
            If True, apply normalization to each frame.

        Sets
        ----
        mov_syn : Tensor
            Video tensor of shape (N, N, 3, nf).
        """
        self.mov_syn = tch.zeros((self.N, self.N, 3, nf), device=self.dev)
        for i in range(nf):
            frame = self.get_frame(adjust=adjust)
            self.mov_syn[:, :, 0, i] = frame
            self.mov_syn[:, :, 1, i] = frame
            self.mov_syn[:, :, 2, i] = frame
