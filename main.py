import cv2 as cv
from lib.common_color import k_mean_cluster

img = cv.imread("img/img_1.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
dim = (500, 300)
# resize image
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

#average_pixel_value(img)
#highest_pixel_frequency(img)
k_mean_cluster(img)

# URL - https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a
