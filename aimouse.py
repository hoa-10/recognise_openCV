import mediapipe as mp
import cv2

cap=cv2.VideoCapture(0)
mphand=mp.solutions.hands
hand=mphand.Hands()
mpdraw=mp.solutions.drawing_utils
finger_coord=[(8,6),(12,10),(16,14),(20,18)]
thumber_coord=(4,2)

while True:
    success,img=cap.read()
    if not success:
        break
    img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hand.process(img_RGB)
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            HandList=[]
            for id, lm in enumerate(handLMS.landmark):
                h, w, c=img.shape
                Cx,Cy= int(lm.x*w), int(lm.y*h)
                HandList.append((Cx,Cy))

            upCount=0
            for coordinate in finger_coord:
                if HandList[coordinate[0]][1] < HandList[coordinate[1]][1]:
                    upCount+=1
            cv2.putText(img,str(upCount),(150,150),cv2.FONT_HERSHEY_PLAIN,10,(0,255,255),3)
            for points in HandList:
                cv2.circle(img,points,10,(0,255,255),3,cv2.FILLED)
            mpdraw.draw_landmarks(img,handLMS,mphand.HAND_CONNECTIONS)
        cv2.imshow("img",img)
        if cv2.waitKey(1) == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()