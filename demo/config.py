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
    'th_start': 0.0,         # Start orientation (degrees)
    'orientation_step': 45.0, # Increment in orientation (degrees)
    'num_clouds': 4,         # Total number of videos to generate
    'duration_sec': 2.0      # Duration of each generated video in seconds
}
