import cv2
import mediapipe as mp
import time
cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mp_draws=mp.solutions.drawing_utils
ptime=0
ctime=0
while True:
    success, img= cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for HandLms in results.multi_hand_landmarks:
            mp_draws.draw_landmarks(img,HandLms,mpHands.HAND_CONNECTIONS)
            for id,lm in enumerate(HandLms.landmark):
            #print(id,lm)
               h, w, c=img.shape
               cx, cy=int(lm.x*w),int(lm.y*h)
               print(id,cx,cy)
               if id==2:
                  cv2.circle(img,(cx,cy),15,(0,255,255),cv2.FILLED)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),3)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()