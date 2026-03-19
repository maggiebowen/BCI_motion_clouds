"""
DynTex: Core class for loading and processing dynamic textures (video sequences)
using PyTorch tensors.

This module provides the DynTex class, which can read video files, perform
color PCA transformations, apply inverse transforms, and compute periodic
component decompositions.
"""

import torch as tch
import imageio as imio
from .utils import periodic_comp


class DynTex:
    """
    DynTex handles loading video data into PyTorch tensors and performing
    color-based PCA and periodic component analysis on the frames.

    Attributes
    ----------
    directory : str
        Path to the folder containing the video file.
    filename : str
        Name of the video file to load (with extension).
    dev : torch.device
        Device on which tensors are allocated (e.g., 'cpu' or 'cuda').
    mov : torch.Tensor
        Video data of shape (ny, nx, 3, nf) after reading.
    ny : int
        Frame height (number of rows).
    nx : int
        Frame width (number of columns).
    nf : int
        Number of frames in the video.
    V : torch.Tensor
        PCA eigenvectors for color decomposition.
    Xm : torch.Tensor
        Mean color values for each frame (3 x nf).
    mov_per : torch.Tensor
        Periodic component representation (complex64) of mov.
    mov_syn : torch.Tensor
        Synthetic color-transformed video (optional).
    """

    def __init__(self, directory, filename, dev=None):
        """
        Initialize the DynTex object.

        Parameters
        ----------
        directory : str
            Directory path where the video file resides.
        filename : str
            Name of the video file (including extension).
        dev : torch.device or None, optional
            Device for tensor storage. If None and CUDA is available,
            uses CUDA, otherwise uses provided device or defaults.
        """
        self.directory = directory
        self.filename = filename
        if dev is None and tch.cuda.is_available():
            self.dev = tch.device('cuda')
        else: 
            self.dev = dev

          # placeholders for later attributes
        self.mov     = None    # video tensor
        self.ny      = None    # height
        self.nx      = None    # width
        self.nf      = None    # frame count
        self.V       = None    # color-PCA eigenvectors
        self.Xm      = None    # per-frame mean colors
        self.mov_per = None    # periodic-component tensor
        self.mov_syn = None    # optional synthetic video
        
    def read(self):
        """
        Read the video file and load frames into a PyTorch tensor.

        Sets the following attributes:
        - mov : Tensor of shape (ny, nx, 3, nf)
        - ny, nx : spatial dimensions of each frame
        - nf : number of frames

        Uses imageio with 'ffmpeg' backend to decode video.
        """
        vid = imio.get_reader(self.directory + self.filename, 'ffmpeg')
        shape = vid.get_meta_data()['size'][::-1] + (3, vid.get_meta_data()['nframes'],)
        self.ny, self.nx, _, self.nf = shape 
        self.mov = tch.empty(shape, device=self.dev)
        i = 0
        for frame in vid.iter_data():
            self.mov[:, :, :, i] = frame
            i += 1
        
    def color_pca(self, n):
        """
        Compute PCA on the color channels of the nth frame.

        Parameters
        ----------
        n : int
            Index of the frame to use for computing the covariance.

        After execution, sets:
        - V : eigenvectors of the 3x3 color covariance matrix
        """
        X = tch.reshape(self.mov[:, :, :, n], (self.ny * self.nx, 3))
        Xm = X.mean(axis=0)
        X = X - Xm
        C = X.T @ X / (X.shape[0] - 1)
        _, _, self.V = tch.svd(C)
    
    def color2pca(self):
        """
        Transform all frames from RGB space to PCA space.

        Uses the eigenvectors V from a prior color_pca call.
        Stores per-frame means in Xm and replaces self.mov with PCA values.
        """
        self.Xm = tch.zeros((3, self.nf), device=self.dev)
        for i in range(self.nf):
            X = tch.reshape(self.mov[:, :, :, i], (self.ny * self.nx, 3))
            self.Xm[:, i] = X.mean(axis=0)
            self.mov[:, :, :, i] = tch.reshape((X - self.Xm[:, i]) @ self.V.T,
                                                (self.ny, self.nx, 3))
            
    def pca2color(self, syn=0):
        """
        Inverse PCA transform back to RGB space.

        Parameters
        ----------
        syn : {0, 1}, optional
            If 0, transform self.mov; if 1, transform self.mov_syn and
            enforce value clipping to [0, 255].
        """
        if syn == 0:
            for i in range(self.nf):
                X = tch.reshape(self.mov[:, :, :, i], (self.ny * self.nx, 3))
                self.mov[:, :, :, i] = tch.reshape(X @ self.V + self.Xm[:, i],
                                                  (self.ny, self.nx, 3))
        elif syn == 1:
            for i in range(self.mov_syn.shape[3]):
                X = tch.reshape(self.mov_syn[:, :, :, i], (self.ny * self.nx, 3)) 
                self.mov_syn[:, :, :, i] = tch.reshape(X @ self.V + self.Xm[:, 0],
                                                      (self.ny, self.nx, 3))
            self.mov_syn[self.mov_syn > 255] = 255  
            self.mov_syn[self.mov_syn < 0] = 0
            
    def per_comp(self):
        """
        Compute the periodic component of each color channel in every frame.

        Uses the periodic_comp utility to transform spatial data into the
        Fourier-based periodic decomposition.

        After execution, sets:
        - mov_per : complex64 tensor matching self.mov shape
        """
        self.mov_per = tch.empty_like(self.mov,
                                      device=self.dev,
                                      dtype=tch.complex64)
        for i in range(self.nf):
            for k in range(3):
                self.mov_per[:, :, k, i] = periodic_comp(self.mov[:, :, k, i])
