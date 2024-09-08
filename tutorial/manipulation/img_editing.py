import cv2 as cv

img = cv.imread('Photos/cyberkitty.jpg')

# Convert to grayscale
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Blur
img = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)

# Edge Cascade
cany = cv.Canny(img, 125, 175)

# Dilating the image
cany = cv.dilate(cany, (7,7), iterations=3)

# Resize
# img = cv.resize(img, (750,500), interpolation=cv.INTER_AREA)

# Crop
# img = img[:500, :500]

cv.imshow('test', img)
cv.imshow('test_canny', cany)

cv.waitKey(0)