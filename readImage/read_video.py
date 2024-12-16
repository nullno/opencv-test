import cv2 as cv

def scale_frame(frame, scale = 1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

capture = cv.VideoCapture('./readImage/assets/robot.mp4')

# cv.CAP_PROP_FRAME_WIDTH = 100

# cv.CAP_PROP_FRAME_HEIGHT = 100

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
   
    # cv.imshow('Robot', frame)

    resized_frame = scale_frame(frame, 0.5)
    cv.imshow("Robot Scale", resized_frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
