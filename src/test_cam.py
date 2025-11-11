import cv2

pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, format=NV12, framerate=30/1 ! "
    "videoconvert ! "
    "appsink drop=true"
)

print("Opening camera...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("? �L�k�}�� Raspberry Pi Camera")
else:
    print("? �۾��w�}�ҡA���bŪ���v��...")
    ret, frame = cap.read()
    print("Frame:", ret, frame.shape if ret else None)
    if ret:
        cv2.imwrite("test_frame.jpg", frame)
    cap.release()
