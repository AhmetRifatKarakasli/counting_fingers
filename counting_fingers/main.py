import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

mphand=mp.solutions.hands
hands=mphand.Hands()
mpdraw=mp.solutions.drawing_utils

tipids=[4,8,12,16,20]
while True:
    _,img=cap.read()
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results=hands.process(imgrgb)
    #print(results.multi_hand_landmarks)

    lmlist=[]
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,handlms,mphand.HAND_CONNECTIONS)

            for id,lm in enumerate(handlms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])

                """if id==8:
                    cv2.circle(img,(cx,cy),9,(255,0,0),cv2.FILLED)"""

    if len(lmlist)!=0:
        fingers=[]
        if lmlist[tipids[0]][1]<lmlist[tipids[0]-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if lmlist[tipids[id]][2]<lmlist[tipids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalf=fingers.count(1)
        print(totalf)
        cv2.putText(img,str(totalf),(30,130),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),8)

    cv2.imshow("Window", img)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



