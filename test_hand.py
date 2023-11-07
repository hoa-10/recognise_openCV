import cv2
import time
import mediapipe as mp
#
cap=cv2.VideoCapture("355102558_6323088411110986_7856285076887942641_n (2).mp4")
pTime=0
mpFacedection=mp.solutions.face_detection
mpdraw=mp.solutions.drawing_utils
facedection=mpFacedection.FaceDetection()
# 
while True:
    success, img= cap.read()
    if not success:
        break
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=facedection.process(imgRGB)
    
    if result.detections:
        for id, detection in enumerate(result.detections):
            bboxC=detection.location_data.relative_bounding_box
            ih , iw, _=img.shape
            bboxC=int(bboxC.xmin*iw),int(bboxC.ymin* ih),int(bboxC.width*iw),int(bboxC.height *ih)
            cv2.rectangle(img,bboxC,(255,0,255),3)

    cTime=time.time()
    fps=1/(cTime -pTime)
    pTime=cTime
    cv2.putText(img,f'FPS :{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow("img",img)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
