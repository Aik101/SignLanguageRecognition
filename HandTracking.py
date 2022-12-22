import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0



while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Изменение цветовой кодировки на RGB, так как модуль hands не поддерживает BGR
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks) #multi_hand_landmarks для того, чтобы камера реагировала на руки

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:   #handLms - Единственная рука
            for id, lm in enumerate(handLms.landmark): #Для получения расположения рук на видео
                #print(id, lm)
                h, w, c = img.shape #Для получения ширины и высоты рук
                cx, cy = int(lm.x*w), int(lm.y*h) #Для получения значений расположения рук в пикселях
                print(id, cx, cy)
                #if id == 4: #Для определенного id
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED) #Для подсвечивания точек

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  #handLms - Рисует точки на руке//mpHands.HAND_CONNECTIONS - рисует соединения между точками

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break