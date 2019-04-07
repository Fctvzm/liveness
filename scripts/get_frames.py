from video_stream_2 import VideoStream
import numpy as np 
import cv2
import time
import os
import uuid
import pandas as pd
import traceback

PATH = '/mnt/SiW_release'
PATH_FACE_DETECTOR = './face_detector'

#face detection net
def get_frames(file_path, type):
	#start video capture
	videoStream = VideoStream(file_path).start()
	time.sleep(1.0)

	start_time = time.time()
	count = 0
	while videoStream.more():
		frame = videoStream.read()
		h, w = frame.shape[:2]
		#face detection
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()

		if len(detections) > 0:
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]
				try:
					face = cv2.resize(face, (256, 256))
					name = str(uuid.uuid4())
					if type == "1":
						type_path = "live"
					elif type == "2":
						type_path = "paper"
					else:
						type_path = "replay"
					file_name = './output/' + type_path + '/' + name + '.png'
					cv2.imwrite(file_name, face)
				except:
					traceback.print_exc()
					pass

	videoStream.stop()


proto_path = os.path.join(PATH_FACE_DETECTOR, 'deploy.prototxt.txt')
model_path = os.path.join(PATH_FACE_DETECTOR, 'res10_300x300_ssd_iter_140000.caffemodel')
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

for root, dirs, files in os.walk(PATH):
	if files:
		for file in files:
			if file.endswith('.mov'):
				print("Processing: {}".format(file))
				type_id = file.split("-")[2] 
				get_frames(os.path.join(root, file), type_id)
