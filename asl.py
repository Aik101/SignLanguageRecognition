import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")


offset = 20
imgSize = 300


folder = "Data/B" #Объявление папки для сохранения данных
counter = 0 #Объявление счетчика

labels = ["A", "B", "C", "fuck", "Spider-Man"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #Белое окно с размерами

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #Матрица с размерами обрезанного окна

        imgCropShape = imgCrop.shape




        aspectRatio = h/w   #Соотношение сторон

        if aspectRatio >=1:
            k = imgSize/h
            wCal = math.ceil(k * w) #Вычисление значений окна/math.ceil - для округления значений
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(img) #Значение для классификации
            print(prediction, index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h) #Вычисление значений окна/math.ceil - для округления значений
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(img) #Значения для классификации

        cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
