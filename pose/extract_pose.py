import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
from utils import load_video

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def extract_pose(
        video_path,
        model_path="models/pose_landmarker_lite.task",
        save_path="data/poses.npy"
):
    """
   Extract pose landmarks from a video using MediaPipe PoseLandmarker.

   Args:
       video_path (str): Path to input video
       model_path (str): Path to .task pose model
       save_path (str): Path to save pose numpy array

   Returns:
       poses (list): List of (33, 2) arrays or None per frame
   """

    frames = load_video(video_path)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )

    poses = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )

            result = landmarker.detect(mp_image)

            if not result.pose_landmarks:
                poses.append(None)
                continue

            landmarks = result.pose_landmarks[0]

            keypoints = []
            for lm in landmarks:
                keypoints.append([lm.x, lm.y])

            poses.append(np.array(keypoints))

    np.save(save_path, np.array(poses, dtype=object))
    print(f"[INFO] Saved poses to {save_path}")

    return poses




