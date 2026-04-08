# Generation dynamical texture
from dyntex.MotionCloud import MotionCloud
from dyntex.DriftingGrating import DriftingGrating
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
import matplotlib as mpl

import torch as tch

import imageio as imio
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import i0, i1
import time
import shutil
import imageio.v2 as imageio

mpl.rcParams['animation.writer'] = 'pillow'
mpl.rcParams['animation.embed_limit'] = 1000

if shutil.which('ffmpeg') is None:
    raise RuntimeError('ffmpeg not found on PATH. Install or add it to PATH.')

# ## Utility functions
# 
def to_numpy(x):
    return x.detach().cpu().numpy()


def write_mp4_safe(mp4_path, frames, fps=30, codec='libx264'):
    frames = np.asarray(frames)
    h, w = frames.shape[1], frames.shape[2]

    def prepare_frame(f):
        f = np.clip(f, 0, 255).astype(np.uint8)
        if f.ndim == 2:
            f = np.stack((f,) * 3, axis=-1)
        elif f.shape[2] == 1:
            f = np.concatenate((f,) * 3, axis=2)
        return f

    try:
        with imageio.get_writer(mp4_path, fps=fps, codec=codec, format='FFMPEG') as writer:
            for f in frames:
                writer.append_data(prepare_frame(f))
    except Exception:
        with imageio.get_writer(mp4_path, fps=fps, codec=codec, format='ffmpeg') as writer:
            for f in frames:
                writer.append_data(prepare_frame(f))

# ## Temporal profile generation
# 
def linear_feature_profile_speed_per_frame(
    num_frames,
    start_frame,
    start_value,
    speed_per_frame,
    variation_duration_frames
):
    num_frames = int(num_frames)
    start_frame = int(start_frame)
    variation_duration_frames = int(variation_duration_frames)
    if variation_duration_frames <= 0:
        raise ValueError('variation_duration_frames must be >0')
    if start_frame < 0:
        raise ValueError('start_frame must be >= 0')
    if start_frame + variation_duration_frames > num_frames:
        raise ValueError('Ramp exceeds total number of frames')

    feature_value = np.full(num_frames, float(start_value), dtype=float)

    # rampe linéaire
    t = np.arange(variation_duration_frames, dtype=float)
    ramp = float(start_value) + float(speed_per_frame) * t

    feature_value[start_frame:start_frame + variation_duration_frames] = ramp

    # valeur stationnaire finale
    final_value = ramp[-1]
    feature_value[start_frame + variation_duration_frames:] = final_value

    return feature_value

# ## Stimulus generation
# 
def generate_nonstationary_MC_second_unit(
    num_frames,
    feature_name,
    feature_start_value,
    feature_speed,
    variation_duration,
    save_npz=False,
    npz_path='stimulus.npz',
    save_mp4=False,
    mp4_path='stimulus.mp4',
):
    params = {
        'sf': 4.0,
        'sf_sig': 1.0,
        'th': 0.0,
        'th_sig': 5.0,
        'tf': 5.0,
        'speed_vec': tch.tensor([0.0, 0.0]),
        'speed_sig': 140.0,
        'octa': 1,
        'N': 256,
        'frame_per_second': 100,
        'start_time': 0.5,
        'time_dt': 0.1,
        'device': 'cuda' if tch.cuda.is_available() else 'cpu',
    }

    # Conversion time per second to time per frame
    fps_native = int(params['frame_per_second'])
    variation_duration_frames = int(np.round(float(variation_duration) * fps_native))
    if variation_duration_frames <= 0:
        raise ValueError('variation_duration (in seconds) is too small')
    if variation_duration_frames > num_frames:
        raise ValueError('variation_duration (in seconds) is longer than the number of frames')
    speed_per_frame = float(feature_speed) / float(fps_native)

    high = params['start_time'] + params['time_dt']
    low = params['start_time'] - params['time_dt']
    start_time_rdm_sec = np.random.uniform(low, high)
    start_frame_rdm = round(start_time_rdm_sec * fps_native)

    print(start_frame_rdm)

    # profil temporel de la feature
    L = linear_feature_profile_speed_per_frame(
        num_frames=num_frames,
        start_frame=start_frame_rdm,
        start_value=feature_start_value,
        speed_per_frame=speed_per_frame,
        variation_duration_frames=variation_duration_frames,
    )
    print(L)

    MC = MotionCloud(
        N=params['N'],
        frame_per_second=fps_native,
        contrast=35,
        ave_lum=127.0,
        over_samp=10,
        verbose=0,
    )
    MC.set_grids()

    frames = tch.zeros((num_frames, params['N'], params['N']), dtype=tch.float32, device=params['device'])
    for i, value in enumerate(L):
        if feature_name == 'sf':
            MC.bandpass_kernel(
                value,
                params['sf_sig'],
                params['th'],
                params['th_sig'],
                params['tf'],
                params['speed_vec'],
                params['octa'],
            )
        elif feature_name == 'th':
            MC.bandpass_kernel(
                params['sf'],
                params['sf_sig'],
                value,
                params['th_sig'],
                params['tf'],
                params['speed_vec'],
                params['octa'],
            )
        else:
            raise ValueError("feature_name must be 'sf' or 'th'")

        frame = MC.get_frame(adjust=True)
        frames[i] = frame

    if isinstance(frames, tch.Tensor):
        frames = frames.detach().cpu().numpy()

    if save_npz:
        np.savez(npz_path, frames=frames, feature_name=feature_name, profile=L, fps=fps_native)

    if save_mp4:
        write_mp4_safe(mp4_path, frames)

    return frames

# ## Optional execution example
# Uncomment and adapt this cell to run a generation example.
# 
# frames = generate_nonstationary_MC_second_unit(
#     num_frames=120,
#     feature_name='sf',
#     feature_start_value=4.0,
#     feature_speed=1.0,
#     variation_duration=0.5,
#     save_npz=False,
#     save_mp4=False,
# )
