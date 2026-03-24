"""
Configuration for generating Motion Cloud stimuli.
All parameters defined here have been checked against the dyntex constraints.
"""

MC_CONFIG = {
    # Initialization params (DynTex/MotionCloud)
    'N': 256,
    'fps': 60,
    'contrast': 35.0,
    'ave_lum': 127.0,
    'over_samp': 2,
    
    # Kernel Generation params (MotionCloud.set_all)
    'kernel_type': 'bandpass',
    'sf': 3.0,          # spatial frequency
    'sf_sig': 1.0,      # spatial frequency bandwidth
    'th_sig': 10.0,     # orientation bandwidth (degrees)
    'tf': 1.0,          # temporal frequency (Hz)
    'spd_scalar': 2.0,  # speed
    'spd_dir': 90.0,    # speed direction
    'octa': 1           # octave flag
}

# Parameters for the generation loop
LOOP_CONFIG = {
    'num_clouds': 6,         # Total number of videos to generate
    'duration_sec': 3.3,     # Duration of each generated video in seconds
    
    # Starting values and step sizes for parameter sweeps
    'th_start': 0.0,
    'th_step': 45.0,
    'sf_start': 1.0,
    'sf_step': 1.0,
    'sf_sig_start': 0.5,
    'sf_sig_step': 0.5,
    'th_sig_start': 5.0,
    'th_sig_step': 5.0,
    'tf_start': 0.5,
    'tf_step': 0.5,
    'spd_scalar_start': 1.0,
    'spd_scalar_step': 1.0,
    'spd_dir_start': 0.0,
    'spd_dir_step': 45.0,
    'contrast_start': 10.0,
    'contrast_step': 10.0,
}
