import numpy as np

def fake_pose_frame(frame):
    """
    returns dummy pose:
    5 keypoints (x, y)
    """

    h, w, _ = frame.shape
    return np.array([
        [w*0.5, h*0.2],
        [w*0.4, h*0.4],
        [w*0.6, h*0.4],
        [w*0.45, h*0.7],
        [w*0.55, h*0.7],
    ])