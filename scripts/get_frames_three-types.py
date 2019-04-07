from video_stream_2 import VideoStream
import traceback
import numpy as np 
import cv2
import time
import os
import uuid
import pandas as pd

PATH = '/mnt/SiW_release'

PATH_FACE_DETECTOR = './face_detector'

SKIP = 5

result = []

modelFile = os.path.join(PATH_FACE_DETECTOR, 'res10_300x300_ssd_iter_140000.caffemodel')
configFile = os.path.join(PATH_FACE_DETECTOR, 'deploy.prototxt.txt')
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def get_face(frame):
	h, w = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104.0, 177.0, 123.0], False, False)
	net.setInput(blob)
	detections = net.forward()

	bboxes = []
	face = None
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.7:
			x1 = int(detections[0, 0, i, 3] * w)
			y1 = int(detections[0, 0, i, 4] * h)
			x2 = int(detections[0, 0, i, 5] * w)
			y2 = int(detections[0, 0, i, 6] * h)
			face = frame[y1:y2, x1:x2]
	return face

def get_frames(file_path, type):
	#start video capture
	videoStream = VideoStream(file_path).start()
	time.sleep(1.0)
	start_time = time.time()
	count = 0
	while videoStream.more():
		frame = videoStream.read()
		if count % SKIP != 0:
			continue
		face = get_face(frame)
		count += 1
		if face is not None:
			try:
				face = cv2.resize(face, (256, 256))
				name = str(uuid.uuid4())
				if type == "1":
					type_path = "live"
				elif type == "2":
					type_path = "paper"
				else:
					type_path = "replay"
				file_name = 'output/' + type_path + '/' + name + '.png'
				cv2.imwrite(file_name, face)
			except Exception:
				traceback.print_exc()
				pass
	videoStream.stop()

for root, dirs, files in os.walk(r'C:\Users\Assem\Desktop\face-spoofing\data\SiW_release'):
	if files:
		for file in files:
			if file.endswith('.mov'):
				print("Processing: {}".format(file))
				type_id = file.split("-")[2]  
				get_frames(os.path.join(root, file), type_id)
