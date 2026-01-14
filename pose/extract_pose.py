import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import loadvideo
from pose_utils import fake_pose_frame

video = loadvideo("C:\\Users\\HP\\Downloads\\testvideo.mp4")

poses = []

for frame in video:
    poses.append(fake_pose_frame(frame))

print(len(poses))
print(poses[0])
