import cv2


name = 'F:/28_Azamat_redmi-s2_webcamHD_1.webm'
cap = cv2.VideoCapture(name)
print(cap.isOpened())