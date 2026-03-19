import numpy as np
import imageio
from dyntex.MotionCloud import MotionCloud

# 1) Initialize MotionCloud
N = 256
fps = 60
MC = MotionCloud(N=N, frame_per_second=fps,
                 contrast=35, ave_lum=127.0,
                 over_samp=2, verbose=0)
MC.set_all(
    kernel_type='bandpass',
    sf=3.0, sf_sig=1.0,
    th=45.0, th_sig=10.0,
    tf=1.0,
    spd_scalar=2.0, spd_dir=90.0,
    octa=1
)
MC.burnout()  # warm-up the AR recursion

# 2) Choose number of frames and output file
n_frames = fps * 5   # 5 seconds
out_path = 'motioncloud.mp4'

# 3) Create video writer (H.264)
writer = imageio.get_writer(
    out_path,
    fps=fps,
    codec='libx264',
    quality=8
)

# 4) Generate frames & write
for i in range(n_frames):
    # advance fourier translation phase
    MC.set_fourier_translation()
    # get one spatial frame (float in [-1…+1] or [0..255] range)
    im = MC.get_frame(adjust=True)  
    # convert to uint8 grayscale
    arr = im.detach().cpu().numpy()
    if arr.max() > 255 or arr.min() < 0:
        arr = np.clip(arr, 0, 255)
    frame_uint8 = arr.astype(np.uint8)
    # imageio wants (H,W) or (H,W,3); here it’s (H,W)
    writer.append_data(frame_uint8)

writer.close()
print(f'Wrote {n_frames} frames → {out_path}')
