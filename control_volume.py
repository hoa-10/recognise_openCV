import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
cap=cv2.VideoCapture(0)
mpHand=mp.solutions.hands
hand=mpHand.Hands()
mpDraw=mp.solutions.drawing_utils

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))

volMin,volMax=volume.GetVolumeRange()[:2]
while True:
    success,img=cap.read()
    if not success:
        break
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hand.process(imgRGB)
    LmList=[]
    if result.multi_hand_landmarks:
        for handlandmark in result.multi_hand_landmarks:
            for id,lm in enumerate(handlandmark.landmark):
                h, w, c=img.shape
                cx, cy=int(lm.x*w),int(lm.y*h)
                LmList.append([id,cx,cy])
            mpDraw.draw_landmarks(img,handlandmark,mpHand.HAND_CONNECTIONS)
            if LmList!= []:
                x1,y1=LmList[4][1],LmList[4][2]
                x2,y2=LmList[8][1],LmList[8][2]
                cv2.circle(img,(x1,y1),15,(0,0,255),3)
                cv2.circle(img,(x2,y2),15,(0,0,255),3)
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),3)
                length=hypot(x2-x1,y2-y1)
                vol=np.interp(length,[15,220],[volMax,volMin])
                print(vol,length)
                volume.SetMasterVolumeLevel(vol,None)
                cv2.imshow("img",img)
                if cv2.waitKey(1) ==ord("q"):
                    break
  
