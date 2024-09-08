import cv2 as cv

img = cv.imread('Photos/cyberkitty.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny = cv.Canny(gray, 125, 175)

countours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(countours)} contour(s) found!')

cv.imshow('edges', canny)

# Treshhold conversion
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('threshold', thresh)

cv.waitKey(0)