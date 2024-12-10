import cv2 as cv
import numpy as np

blank = np.zeros((300, 500, 3), dtype="uint8")

# draw a rectangle
blank[0:200, 0:300] = 0, 255, 0  # red


# draw a line

lineStart = (0, 0)  # top left
lineEnd = (300, 200)  # bottom right
color = (0, 0, 255)  # line color
thickness = 3
cv.line(blank, lineStart, lineEnd, color, thickness)
cv.imshow("Line", blank)


cv.waitKey(0)
