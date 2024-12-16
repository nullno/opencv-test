import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
blank = np.zeros((400, 600, 3), dtype="uint8")

# draw a rectangle
blank[0:200, 0:300] = 0, 255, 0  # red


# draw a line

lineStart = (200, 300)  # top left
lineEnd = (300, 0)  # bottom right
color = (0, 0, 255)  # line color
thickness = 3
cv.line(blank, lineStart, lineEnd, color, thickness)
cv.imshow("Line", blank)


cv.waitKey(0)


