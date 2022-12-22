# -*- coding: utf-8 -*-
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300


folder = "Data/O" #Объявление папки для сохранения данных
counter = 0 #Объявление счетчика


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox'] #Ограничивающая рамка на основе высоты и ширины

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #Создание белой ограничивающей рамки посредством матрицы и выставление размера, кодировки а также диапазона цветов жестов внутри белой рамки, чтобы получить правильные цвета

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #Размеры ограничивающей рамки с учетом отступов

        imgCropShape = imgCrop.shape




        aspectRatio = h/w

        if aspectRatio >=1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1 #Счетчик снимков
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  #Для сохранения данных
        print(counter) #Вывод значения счетчика