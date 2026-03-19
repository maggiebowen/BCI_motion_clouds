import numpy as np
from torch.cuda import is_available
from dyntex.MotionCloud import MotionCloud

from psychopy import event, core, visual
from psychopy.visual import Slider

# ——————————————————————————————————————————————
# 1) Set up MotionClouds
# ——————————————————————————————————————————————
dev = 'cuda' if is_available() else 'cpu'
N, fps = 256, 60

MC1 = MotionCloud(N=N, frame_per_second=fps,
                  contrast=35, ave_lum=127.0,
                  over_samp=4, verbose=0, dev=dev)
MC2 = MotionCloud(N=N, frame_per_second=fps,
                  contrast=35, ave_lum=127.0,
                  over_samp=4, verbose=0, dev=dev)

choose_kernel = 'bandpass'
tf, octa = 4.0, 1

# Default params for MC1 and MC2
defaults1 = dict(sf=1.0,   sf_sig=2.5, th=0.0,  th_sig=10.0, 
                spd_scalar=1.0, spd_dir=0.0)
defaults2 = dict(sf=6.0,   sf_sig=0.5, th=0.0,  th_sig=5.0,  
                spd_scalar=1.0, spd_dir=0.0)

# “Burn in” both
MC1.set_all(choose_kernel, **defaults1, tf=tf, octa=octa); MC1.burnout()
MC2.set_all(choose_kernel, **defaults2, tf=tf, octa=octa); MC2.burnout()

# ——————————————————————————————————————————————
# 2) Create PsychoPy window
# ——————————————————————————————————————————————
scale = 2
win = visual.Window([scale*500, 500], color='gray', units='norm')

# Prepare the ImageStim (initial blank)
blank = np.zeros((N, N), dtype=np.float32)
stim = visual.ImageStim(win, image=blank, interpolate=False,
                        size=(0.8, scale*0.8), pos=(0, -0.1))

# ——————————————————————————————————————————————
# 3) Composite slider (top center)
# ——————————————————————————————————————————————
labelHeight = 0.05
comp_label  = visual.TextStim(win, text='Composite Percent', pos=(0, +0.92),
                              height=0.07, color='white')
comp_slider = Slider(win,
                     ticks=(0, 100),
                     labels=['100% MC1', '100% MC2'],
                     pos=(0, +0.8),
                     size=(1.6, 0.07),
                     granularity=0.5,
                     style='slider',
                     color='white',
                     fillColor='black',
                     borderColor='white',
                     markerColor='dodgerblue',
                     font='Arial',
                     startValue=50.0,
                     labelHeight=labelHeight)
comp_slider.setRating(50.0)

# ——————————————————————————————————————————————
# 4) Play / Pause / Reset buttons
# ——————————————————————————————————————————————
mouse    = event.Mouse(win=win)
btn_w, btn_h = 0.15, 0.07
btn_y = +0.925

play_btn  = visual.Rect(win, width=btn_w, height=btn_h, pos=(-0.7, btn_y),
                        fillColor='darkgray', lineColor='white')
pause_btn = visual.Rect(win, width=btn_w, height=btn_h, pos=(-0.5, btn_y),
                        fillColor='darkgray', lineColor='white')
reset_btn = visual.Rect(win, width=btn_w, height=btn_h, pos=(-0.3, btn_y),
                        fillColor='darkgray',   lineColor='white')

play_txt  = visual.TextStim(win, text='Play',  pos=(-0.7, btn_y), height=0.04, color='white')
pause_txt = visual.TextStim(win, text='Pause', pos=(-0.5, btn_y), height=0.04, color='white')
reset_txt = visual.TextStim(win, text='Reset', pos=(-0.3, btn_y), height=0.04, color='white')

playing = True
# store last frame so Pause still shows something
last_image = blank.copy()

# ——————————————————————————————————————————————
# 5) Define MC1 sliders & labels (left column)
# ——————————————————————————————————————————————
ys = [+0.55, +0.3, +0.05, -0.2, -0.45, -0.7]
dys = 0.08
keys = ['sf', 'sf_sig', 'th', 'th_sig', 'spd_scalar', 'spd_dir']
ranges = [(0.5,7.5),(0.1,5),(0,180),(1,50),(0,20),(0,360)]
ticks_labels = [['0.5','7.5'], ['0.1','5'], ['0','180'], ['1','50'], ['0','20'], ['0','360']]
granularities = [0.01, 0.1, 1, 0.1, 0.1, 1]

mc1_sliders = {}
mc1_labels  = {}

for k, y, rng, labs, gran in zip(keys, ys, ranges, ticks_labels, granularities):
    sld = Slider(win,
                 ticks=rng, labels=labs,
                 pos=(-0.7, y), size=(0.4, 0.07),
                 granularity=gran, style='slider',
                 color='white', fillColor='black',
                 borderColor='white', markerColor='dodgerblue',
                 font='Arial', startValue=defaults1[k],
                 labelHeight=labelHeight)
    sld.setRating(defaults1[k])
    mc1_sliders[k] = sld

    txt = visual.TextStim(win,
                          text=f"MC1 – {k.replace('_',' ').upper()}",
                          pos=(-0.7, y+dys),
                          height=0.05, color='white')
    mc1_labels[k] = txt

# ——————————————————————————————————————————————
# 6) Define MC2 sliders & labels (right column)
# ——————————————————————————————————————————————
mc2_sliders = {}
mc2_labels  = {}

for k, y, rng, labs, gran in zip(keys, ys, ranges, ticks_labels, granularities):
    sld = Slider(win,
                 ticks=rng, labels=labs,
                 pos=(+0.7, y), size=(0.4, 0.07),
                 granularity=gran, style='slider',
                 color='white', fillColor='black',
                 borderColor='white', markerColor='dodgerblue',
                 font='Arial', startValue=defaults2[k],
                 labelHeight=labelHeight)
    sld.setRating(defaults2[k])
    mc2_sliders[k] = sld

    txt = visual.TextStim(win,
                          text=f"MC2 – {k.replace('_',' ').upper()}",
                          pos=(+0.7, y+dys),
                          height=0.05, color='white')
    mc2_labels[k] = txt

# ——————————————————————————————————————————————
# 7) Main loop
# ——————————————————————————————————————————————
clock = core.Clock()
while True:
    # -- handle button clicks once per frame --
    if mouse.getPressed()[0]:
        if play_btn.contains(mouse):
            playing = True
        elif pause_btn.contains(mouse):
            playing = False
        elif reset_btn.contains(mouse):
            playing = False
            comp_slider.setRating(50.0)
            for k in keys:
                mc1_sliders[k].setRating(defaults1[k])
                mc2_sliders[k].setRating(defaults2[k])

    # 1) read slider values
    comp_ratio = comp_slider.getRating() / 100.0
    vals1 = {k: mc1_sliders[k].getRating() for k in keys}
    vals2 = {k: mc2_sliders[k].getRating() for k in keys}

    # 2) update & grab new frames only if playing
    if playing:
        MC1.set_all(choose_kernel,
                    vals1['sf'], vals1['sf_sig'],
                    vals1['th'], vals1['th_sig'],
                    tf,
                    vals1['spd_scalar'], vals1['spd_dir'],
                    octa)
        MC1.set_fourier_translation()
        im1 = MC1.get_frame(True).detach().cpu().numpy()

        MC2.set_all(choose_kernel,
                    vals2['sf'], vals2['sf_sig'],
                    vals2['th'], vals2['th_sig'],
                    tf,
                    vals2['spd_scalar'], vals2['spd_dir'],
                    octa)
        MC2.set_fourier_translation()
        im2 = MC2.get_frame(True).detach().cpu().numpy()

        # blend & store last_image
        blended = (1-comp_ratio)*im1 + comp_ratio*im2
        last_image = np.clip(blended/128.0 - 1, -1, 1)

    # 3) draw the (possibly frozen) image
    stim.image = last_image
    stim.draw()

    # 4) draw all UI
    comp_label.draw();  comp_slider.draw()
    play_btn.draw();    play_txt.draw()
    pause_btn.draw();   pause_txt.draw()
    reset_btn.draw();   reset_txt.draw()

    for k in keys:
        mc1_labels[k].draw(); mc1_sliders[k].draw()
        mc2_labels[k].draw(); mc2_sliders[k].draw()

    win.flip()

    # 5) exit on ESC
    if event.getKeys(['escape']):
        break

    # 6) throttle
    core.wait(max(1/60 - clock.getTime(), 0))
    clock.reset()

win.close()
core.quit()
