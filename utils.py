import cv2

def loadvideo(path):
    vidcap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = vidcap.read()

        if not ret:
            break
        frames.append(frame)

    vidcap.release()
    return frames