import cv2 as cv
def scale_image(img, scale = 1):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


img = cv.imread("./assets/qiqiu.jpg")

cv.imshow("Img", img)

resized_img = scale_image(img, 0.8)

cv.imshow("Img Scale", resized_img)

cv.waitKey(0)
