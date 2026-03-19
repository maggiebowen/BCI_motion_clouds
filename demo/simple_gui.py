import numpy as np
from torch.cuda import is_available
from dyntex.MotionCloud import MotionCloud

from psychopy import event, core, visual
from psychopy.visual import Slider

# —————————————————————————————————————————————————————————
# 1) Setup MC & window
# —————————————————————————————————————————————————————————
dev = 'cuda' if is_available() else 'cpu'
MC = MotionCloud(N=256, frame_per_second=60,
                 contrast=35, ave_lum=127.0,
                 over_samp=4, verbose=0, dev=dev)
choose_kernel = 'bandpass'
tf, octa = 4.0, 1

w, scale = 400, 1.5
win = visual.Window([w, scale*w], color='gray', units='norm')

# initial defaults
defaults = dict(
    sf=2.0, th=0.0, spd=1.0,
    sf_sig=2.5, th_sig=10.0, spd_dir=0.0
)
MC.set_all(choose_kernel,
           defaults['sf'],    defaults['sf_sig'],
           defaults['th'],    defaults['th_sig'],
           tf,
           defaults['spd'],   defaults['spd_dir'],
           octa)
MC.burnout()

# —————————————————————————————————————————————————————————
# 2) Sliders (unchanged)
# —————————————————————————————————————————————————————————
slider_w, slider_h = 0.8, 0.075

sf_slider = Slider(win, ticks=(0.5,7.5), labels=['0.5','7.5'],
                   pos=(-0.5, -0.85), size=(slider_w, slider_h),
                   granularity=0.01, startValue=defaults['sf'],
                   style='slider', color='white', fillColor='black',
                   borderColor='white', markerColor='dodgerblue')
sf_slider.setRating(defaults['sf'])

th_slider = Slider(win, ticks=(0,180), labels=['0','180'],
                   pos=(-0.5, -0.625), size=(slider_w, slider_h),
                   granularity=1, startValue=defaults['th'],
                   style='slider', color='white', fillColor='black',
                   borderColor='white', markerColor='dodgerblue')
th_slider.setRating(defaults['th'])

spd_slider = Slider(win, ticks=(0,20), labels=['0','20'],
                    pos=(-0.5, -0.40), size=(slider_w, slider_h),
                    granularity=0.1, startValue=defaults['spd'],
                    style='slider', color='white', fillColor='black',
                    borderColor='white', markerColor='dodgerblue')
spd_slider.setRating(defaults['spd'])

sf_sig_slider = Slider(win, ticks=(0.1,5), labels=['0.1','5'],
                       pos=(0.5, -0.85), size=(slider_w,slider_h),
                       granularity=0.1, startValue=defaults['sf_sig'],
                       style='slider', color='white', fillColor='black',
                       borderColor='white', markerColor='dodgerblue')
sf_sig_slider.setRating(defaults['sf_sig'])

th_sig_slider = Slider(win, ticks=(1,50), labels=['1','50'],
                       pos=(0.5,  -0.625), size=(slider_w,slider_h),
                       granularity=0.1, startValue=defaults['th_sig'],
                       style='slider', color='white', fillColor='black',
                       borderColor='white', markerColor='dodgerblue')
th_sig_slider.setRating(defaults['th_sig'])

spd_dir_slider = Slider(win, ticks=(0,360), labels=['0','360'],
                        pos=(0.5, -0.40), size=(slider_w,slider_h),
                        granularity=1, startValue=defaults['spd_dir'],
                        style='slider', color='white', fillColor='black',
                        borderColor='white', markerColor='dodgerblue')
spd_dir_slider.setRating(defaults['spd_dir'])

# labels
sf_label      = visual.TextStim(win, text='Spatial Frequency', pos=(-0.5,-0.75), height=0.05, color='white')
th_label      = visual.TextStim(win, text='Orientation (°)',   pos=(-0.5,-0.525), height=0.05, color='white')
spd_label     = visual.TextStim(win, text='Speed',             pos=(-0.5,-0.300), height=0.05, color='white')
sf_sig_label  = visual.TextStim(win, text='SF Sigma',          pos=( 0.5,-0.75),  height=0.05, color='white')
th_sig_label  = visual.TextStim(win, text='Theta Sigma (°)',   pos=( 0.5,-0.525), height=0.05, color='white')
spd_dir_label = visual.TextStim(win, text='Speed Dir (°)',     pos=( 0.5,-0.300), height=0.05, color='white')

# —————————————————————————————————————————————————————————
# 3) Buttons setup
# —————————————————————————————————————————————————————————
mouse = event.Mouse(win=win)
btn_w, btn_h = 0.25, 0.1
btn_y = +0.9

play_btn  = visual.Rect(win, width=btn_w, height=btn_h, pos=(-0.5, btn_y),
                        fillColor='darkgray', lineColor='white')
pause_btn = visual.Rect(win, width=btn_w, height=btn_h, pos=(   0, btn_y),
                        fillColor='darkgray', lineColor='white')
reset_btn = visual.Rect(win, width=btn_w, height=btn_h, pos=(+0.5, btn_y),
                        fillColor='darkgray',   lineColor='white')

play_txt  = visual.TextStim(win, text='Play',  pos=(-0.5, btn_y), height=0.04, color='white')
pause_txt = visual.TextStim(win, text='Pause', pos=(   0, btn_y), height=0.04, color='white')
reset_txt = visual.TextStim(win, text='Reset', pos=(+0.5, btn_y), height=0.04, color='white')

playing = True
# prepare first frame
last_im = np.clip(MC.get_frame(True).detach().cpu().numpy()/128.0 - 1, -1, 1)
stim = visual.ImageStim(win, image=last_im, interpolate=False,
                        size=(scale*0.9, 0.9), pos=(0, +0.3))

# —————————————————————————————————————————————————————————
# 4) Main loop
# —————————————————————————————————————————————————————————
clock = core.Clock()
while True:
    # handle button clicks
    if mouse.getPressed()[0]:
        if play_btn.contains(mouse):
            playing = True
        elif pause_btn.contains(mouse):
            playing = False
        elif reset_btn.contains(mouse):
            playing = False
            # reset sliders
            for sld, val in [
                (sf_slider,    defaults['sf']),
                (th_slider,    defaults['th']),
                (spd_slider,   defaults['spd']),
                (sf_sig_slider,defaults['sf_sig']),
                (th_sig_slider,defaults['th_sig']),
                (spd_dir_slider,defaults['spd_dir'])
            ]:
                sld.setRating(val)

    # read slider values
    sf     = sf_slider.getRating()
    th     = th_slider.getRating()
    spd    = spd_slider.getRating()
    sf_sig = sf_sig_slider.getRating()
    th_sig = th_sig_slider.getRating()
    spd_dir= spd_dir_slider.getRating()

    # update MC & image only if playing
    if playing:
        MC.set_all(choose_kernel,
                   sf,    sf_sig,
                   th,    th_sig,
                   tf,
                   spd,   spd_dir,
                   octa)
        MC.set_fourier_translation()
        last_im = np.clip(MC.get_frame(True).detach().cpu().numpy()/128.0 - 1, -1, 1)
        stim.image = last_im

    # --- draw everything ---
    stim.draw()
    # sliders + labels
    for lbl, sld in [
        (sf_label, sf_slider), (th_label, th_slider), (spd_label, spd_slider),
        (sf_sig_label, sf_sig_slider), (th_sig_label, th_sig_slider), (spd_dir_label, spd_dir_slider)
    ]:
        lbl.draw(); sld.draw()

    # buttons + labels
    for btn, txt in [
        (play_btn, play_txt), (pause_btn, pause_txt), (reset_btn, reset_txt)
    ]:
        btn.draw(); txt.draw()

    win.flip()

    # exit on ESC
    if event.getKeys(['escape']):
        break

    # cap to ~60 Hz
    core.wait(max(1/60 - clock.getTime(), 0))
    clock.reset()

win.close()
core.quit()
