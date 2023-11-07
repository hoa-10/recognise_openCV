import mediapipe as mp
import cv2 
import time
import hand_tracking as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0) 

cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector=htm.handDetector()
while True:
    success, img = cap.read()
    if not success:
        break
    img=detector.findHands(img)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS:{int(fps)}", (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("img", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()