import cv2 as cv

vid = cv.VideoCapture('Videos/dog.mp4')
while True:
    isTrue, frame = vid.read()

    if not isTrue:
        break

    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

vid.release()
cv.destroyAllWindows()

cv.waitKey(0)