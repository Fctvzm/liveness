import uuid
import cv2
import numpy as np 
import pandas as pd
import os
from mtcnn.mtcnn import MTCNN

CLASS = 0
TYPE = None
PRINTED_TYPE = None
REPLAYED = None
detector = MTCNN()

def get_info(file_name):
	return file_name.split('_')

def get_frames(video):
	# info = get_info(video_name)
	# subject_name = info[1]
	# captured_device = info[2]

	capture = cv2.VideoCapture(video)
	n_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

	if capture.isOpened():
		pos_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
		fps = int(capture.get(cv2.CAP_PROP_FPS))
		while True:
			ret, frame = capture.read()
			if ret:
				result = detector.detect_faces(frame)
				if (len(result) > 0):
					bndbox = result[0]['box']

					cv2.rectangle(frame, (bndbox[0], bndbox[1]), (bndbox[0]+bndbox[2], bndbox[1] + bndbox[3]), 
						(0,155,255), 2)

					cv2.imshow("frame", frame)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
			if pos_frame == n_frames:
				break
			elif ret == False:
				break
			pos_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
	capture.release()

get_frames('F:/spoof-faces/Webcam/6_Roman_redmi-s2_webcamHD_1.webm')