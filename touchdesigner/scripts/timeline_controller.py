"""
TouchDesigner Timeline Controller & Cue Dispatcher for MotionVision TD
Executes storyboard transitions and drives visual parameters based on:
- Sequence 1: 6-scene storyboard (cinematics/scenes.json)
- Sequence 2: 5-act storyboard (cinematics/scenes_sequence2.json)
"""

import json

def onFrameChange(channelValue, prevChannelValue):
    try:
        import td
    except ImportError:
        return

    curr_frame = int(channelValue)

    ui = td.op('/project1/ui_controls')
    seq_choice = getattr(ui.par, 'Sequence', 'sequence2') if ui else 'sequence2'

    compositor = td.op('/project1/compositor')
    switch_mode = compositor.op('switch_modes') if compositor else None

    if seq_choice == 'sequence2':
        # Sequence 2 (5-Act Concept):
        # Act 1: 0 - 300   (0 - 10s)  Original footage & player tracking
        # Act 2: 300 - 750 (10 - 25s) 3D Motion reconstruction (Gold & Red meshes)
        # Act 3: 750 - 1200 (25 - 40s) Dribbling intelligence (Skeleton & vectors)
        # Act 4: 1200 - 1500 (40 - 50s) Particle transformation (Particles + RGB-D)
        # Act 5: 1500 - 1800 (50 - 60s) Cinematic finale (Multi-layer goal & ovation)
        if curr_frame < 300:
            active_mode = 0 # Original + Tracking
        elif curr_frame < 750:
            active_mode = 1 # Colored Meshes
        elif curr_frame < 1200:
            active_mode = 2 # 3D Skeleton & Biomechanics
        elif curr_frame < 1500:
            active_mode = 3 # Particle & RGB-D
        else:
            active_mode = 4 # Cinematic Finale Tri-Morph
    else:
        # Sequence 1 (6-Scene Storyboard)
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
