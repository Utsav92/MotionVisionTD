"""
TouchDesigner Offline Movie File Out Stepping Engine
Renders the complete 60-second video deterministically frame-by-frame.
"""

def render_movie(output_path="renders/final/motionvision_ronaldinho_60s.mp4", total_frames=1800, fps=30):
    try:
        import td
    except ImportError:
        print("[TD EXPORT] Error: This script must be run inside TouchDesigner or TouchEngine.")
        return False

    print(f"[TD EXPORT] Starting deterministic offline render: {total_frames} frames -> {output_path}")

    # Set real-time flag to false for deterministic frame cooking
    td.project.realTime = False
    td.project.cookRate = fps

    mov_out = td.op('/project1/export_manager/moviefileout1')
    if mov_out is None:
        print("[TD EXPORT] moviefileout1 operator not found!")
        return False

    mov_out.par.file = output_path
    mov_out.par.record = 1

    for frame in range(total_frames):
        td.timeline.frame = frame
        td.project.cook()

    mov_out.par.record = 0
    td.project.realTime = True
    print(f"[TD EXPORT] Render complete: {output_path}")
    return True
