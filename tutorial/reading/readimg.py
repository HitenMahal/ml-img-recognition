import cv2 as cv

img = cv.imread('Photos/cyberkitty.jpg')
cv.imshow('test', img)

cv.waitKey(0)