import cv2 as cv
from transformers import pipeline
import mediapipe as mp

RES = (520*2, 293*2)
WEB_CAM_ID = 1 # Device id of the video input device

# vid = cv.VideoCapture("Videos/group.mp4")
vid = cv.VideoCapture(WEB_CAM_ID) 
cTime, pTime = 0, 0

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection.FaceDetection()

while True:
    cTime = cv.getTickCount()
    fps = cv.getTickFrequency() / (cTime - pTime)
    pTime = cTime

    isTrue, frame = vid.read()
    if not isTrue:
        break

    frame = cv.resize(frame, RES)
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    framePainted = frame.copy()

    faces = mpFaceDetection.process(frameRGB)

    if faces.detections:
        for id, detection in enumerate(faces.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = framePainted.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            cv.rectangle(framePainted, bbox, (255, 0, 255), 2)

            closeupBox = [bbox[0]-50, bbox[1]-50, bbox[2]+100, bbox[3]+100]
            if bbox[0]-50 > 0 and bbox[1]-50 > 0:
                closeup = frame[closeupBox[1]:closeupBox[1]+closeupBox[3], closeupBox[0]:closeupBox[0]+closeupBox[2]]
                cv.imshow(f'Closeup_{id}', closeup)
    
    cv.putText(framePainted, f'FPS: {int(fps)}', (10, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    cv.imshow('Video', framePainted)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break