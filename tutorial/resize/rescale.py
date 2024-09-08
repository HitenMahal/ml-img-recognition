import cv2 as cv

img = cv.imread('Photos/cyberkitty.jpg')
vid = cv.VideoCapture('Videos/dog.mp4')

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


half_img = rescaleFrame(img, 0.5)

cv.imshow('test', img)
cv.imshow('test_half', half_img)

cv.waitKey(0)