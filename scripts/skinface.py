# -*- coding: utf-8 -*-
import cv2
import numpy as np
import imutils
import uuid

cap = cv2.VideoCapture(0)
cv2.startWindowThread()

def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    return faces

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
clas = 0
while(True):
    ret, frame = cap.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect_face(img_gray, faceCascade)

    # x_left, x_right = (200, 550)
    # y_left, y_right = (200, 550)
    # rois = frame[y_left:y_right, x_left:x_right]

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 2)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = mask)

    sum = 0
    for i, (x, y, w, h) in enumerate(faces):
    
        # roi = frame[y:y+h, x:x+w]
        
        # cv2.rectangle(frame, (x_left, y_left), (x_right, y_right), (0, 255, 0), 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(skin, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        # upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        # mask = cv2.erode(mask, kernel, iterations = 2)
        # mask = cv2.dilate(mask, kernel, iterations = 2)

        # mask = cv2.GaussianBlur(mask, (3, 3), 0)
        # skin = cv2.bitwise_and(roi, roi, mask = mask)
        gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]
        size = w*h
        for i in range(len(roi)):
            for j in range(len(roi[i])):
                if roi[i][j] > 0:
                    sum += 1
        point = (x, y-5)
        perc = 0.3*size
        if sum > perc:
            text = 'True'
            print('size: ', size, 'sum: ', sum, ' ', text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img=frame, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img=skin, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            text = 'False'
            print('size: ', size, 'sum: ', sum, ' ', text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img=frame, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img=skin, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            
        cv2.imshow('frame', frame)
        cv2.imshow('skin', skin)
        # cv2.imshow('gray', gray)
    
    # nam = str(uuid.uuid4()) + '.jpg'
    # name = 'faces/' + str(clas) + '/' + nam
    # cv2.imwrite(name, gray)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()