"""
TouchDesigner Script CHOP Callbacks for MotionVision TD
Reads binary motion_data.npz and outputs frame-accurate multi-person 3D channels.
"""

import numpy as np

def onCook(scriptOp):
    scriptOp.clear()

    # Get target file from parent or default path
    npz_path = "data/poses/motion_data.npz"
    try:
        data = np.load(npz_path)
    except Exception as e:
        scriptOp.appendChan('error')
        scriptOp['error'].vals = [1.0]
        return

    # Total frames
    total_frames = int(data.get('total_frames', 1800))
    scriptOp.numSamples = total_frames

    # Extract all person keys
    keys = list(data.keys())
    person_ids = set()
    for k in keys:
        if k.startswith('person_') and '_frames' in k:
            pid = k.split('_')[1]
            person_ids.add(pid)

    for pid in sorted(person_ids):
        prefix = f"person_{pid}_"
        trans_key = prefix + "root_trans"
        orient_key = prefix + "root_orient"
        conf_key = prefix + "confidence"

        if trans_key in data:
            trans = data[trans_key] # (N, 3)
            cx = scriptOp.appendChan(f"p{pid}_tx")
            cy = scriptOp.appendChan(f"p{pid}_ty")
            cz = scriptOp.appendChan(f"p{pid}_tz")
            cx.vals = trans[:, 0].tolist()
            cy.vals = trans[:, 1].tolist()
            cz.vals = trans[:, 2].tolist()

        if orient_key in data:
            orient = data[orient_key] # (N, 3)
            rx = scriptOp.appendChan(f"p{pid}_rx")
            ry = scriptOp.appendChan(f"p{pid}_ry")
            rz = scriptOp.appendChan(f"p{pid}_rz")
            rx.vals = orient[:, 0].tolist()
            ry.vals = orient[:, 1].tolist()
            rz.vals = orient[:, 2].tolist()

        if conf_key in data:
            conf = data[conf_key] # (N,)
            c_chan = scriptOp.appendChan(f"p{pid}_conf")
            c_chan.vals = conf.tolist()
