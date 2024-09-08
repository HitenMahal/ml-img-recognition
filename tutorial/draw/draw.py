import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
img = cv.imread('Photos/cyberkitty.jpg')

# Paint entire image
img[:] = 255, 0, 0

# Paint portion
img[100:200, 300:400] = 0, 255, 0

# Draw rectangle
cv.rectangle(img, (0,0), (250, 250), (0, 0, 255), thickness=cv.FILLED)

# Draw circle
cv.circle(img, (300,300), 50, (255, 255, 0), thickness=4)

# Draw line
cv.line(img, (400,400), (700,400), (255, 0, 255), thickness=10)

# Write text
cv.putText(img, "HELLO,WORLD!", (350,350), cv.FONT_HERSHEY_TRIPLEX, 1.0, (100,255,255), thickness=2)

cv.imshow('test', img)
cv.waitKey(0)