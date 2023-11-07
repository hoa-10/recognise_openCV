import mediapipe as mp
import cv2
import time
mpPose=mp.solutions.pose
mpdraw=mp.solutions.drawing_utils
pose=mpPose.Pose()
cap=cv2.VideoCapture("360017571_6150158015082884_6412724758783902614_n.mp4")
Ptime=0
Ctime=0
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=pose.process(imgRGB)
    if result.pose_landmarks:
        mpdraw.draw_landmarks(img,result.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(result.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),2,(0,255,255),cv2.FILLED)
    
    Ctime=time.time()
    fps=1/(Ctime-Ptime)
    ptime=Ctime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow("img",img)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
