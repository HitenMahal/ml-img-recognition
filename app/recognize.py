import cv2 as cv
from transformers import pipeline
import mediapipe as mp
from PIL import Image
import multiprocessing 
from collections import defaultdict

RES = (520*2, 293*2)
CLOSEUP_RES = (512,512)
CLOSEUP_MARGIN = 100
WEB_CAM_ID = 1 # Device id of the video input deviced
GPU_ID = -1 # -1 for CPU, 0 for GPU

def identity_worker(ids_of_interest, face_id_to_identity, face_id_to_closeup):
    pipe = pipeline("image-classification", model="cledoux42/Ethnicity_Test_v003", device=GPU_ID)

    while True:
        for id in ids_of_interest:
            closeup = face_id_to_closeup.get(id, None)
            if closeup is not None:
                img = Image.fromarray(closeup)
                # img.show()
                res = pipe(img)
                face_id_to_identity[id] = res[0]['label']

def paint_face_box(framePainted, bbox, id, identity):
    cv.rectangle(framePainted, bbox, (255, 0, 255), 2)
    cv.putText(framePainted, f'ID: {id}, Identity: {identity}', (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    return framePainted


def main():
    # vid = cv.VideoCapture("Videos/students.mp4")
    vid = cv.VideoCapture(WEB_CAM_ID) 
    cTime, pTime = 0, 0

    mpDraw = mp.solutions.drawing_utils
    mpFaceDetection = mp.solutions.face_detection.FaceDetection()

    manager = multiprocessing.Manager()
    face_id_to_identity = manager.dict()
    face_id_to_closeup = manager.dict()
    ids_of_interest = manager.list()

    worker = multiprocessing.Process(target=identity_worker, args=(ids_of_interest,face_id_to_identity,face_id_to_closeup))
    worker.start()

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

        newIds = []
        if faces.detections:
            for id, detection in enumerate(faces.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = framePainted.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)


                framePainted = paint_face_box(framePainted, bbox, id, face_id_to_identity.get(id, None))

                closeupBox = [bbox[0]-CLOSEUP_MARGIN, bbox[1]-CLOSEUP_MARGIN, bbox[2]+CLOSEUP_MARGIN*2, bbox[3]+CLOSEUP_MARGIN*2]
                if bbox[0]-CLOSEUP_MARGIN > 0 and bbox[1]-CLOSEUP_MARGIN > 0:
                    closeup = frameRGB[closeupBox[1]:closeupBox[1]+closeupBox[3], closeupBox[0]:closeupBox[0]+closeupBox[2]]
                    face_id_to_closeup[id] = cv.resize(closeup, CLOSEUP_RES)
                    # cv.imshow(f'Closeup_{id}', face_id_to_closeup[id])
                newIds.append(id)
        ids_of_interest[:] = newIds

        cv.putText(framePainted, f'FPS: {int(fps)}', (10, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv.imshow('Video', framePainted)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break
    
    worker.terminate()
    worker.join()
    return

if __name__ == "__main__":
    main()