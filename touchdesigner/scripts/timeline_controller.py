"""
TouchDesigner Timeline Controller & Cue Dispatcher for MotionVision TD
Executes storyboard transitions and drives visual parameters based on cinematics/scenes.json.
"""

import json

def onFrameChange(channelValue, prevChannelValue):
    try:
        import td
    except ImportError:
        return

    curr_frame = int(channelValue)

    # Determine active scene (60s @ 30fps)
    # Scene 1: 0 - 240
    # Scene 2: 240 - 540
    # Scene 3: 540 - 900
    # Scene 4: 900 - 1200
    # Scene 5: 1200 - 1500
    # Scene 6: 1500 - 1800

    compositor = td.op('/project1/compositor')
    switch_mode = compositor.op('switch_modes') if compositor else None

    if curr_frame < 240:
        active_mode = 0 # Mode A (Intro)
    elif curr_frame < 540:
        active_mode = 0 # Mode A (Meshes)
    elif curr_frame < 900:
        active_mode = 1 # Mode B (Skeleton)
    elif curr_frame < 1200:
        active_mode = 2 # Mode C (Particles)
    elif curr_frame < 1500:
        active_mode = 5 # Mode F (Analytics)
    else:
        active_mode = 3 # Mode D (RGB-D Hologram)

    if switch_mode:
        switch_mode.par.index = active_mode
