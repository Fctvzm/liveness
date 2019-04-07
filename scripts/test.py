import uuid
import cv2
import numpy as np 
import pandas as pd
import os
from face_detector import Face

cls = None
spoof_type = None
printed_type = None
replayed_device = None
captured_device = None
ROOT = './data'
DIR_NAME = './images'
face_model = Face()


def get_info(file_name):
	return file_name.split('_')

def get_frames(video, video_name, result, cls):
	info = get_info(video_name)
	if cls == 1:
		if (info[2] in ['glossy', 'mat']):
			spoof_type = 'printed'
			printed_type = info[2]
			captured_device = info[3]
			replayed_device = None
		else:
			spoof_type = 'replayed'
			captured_device = info[2]
			replayed_device = info[3]
			printed_type = None
	else:
		spoof_type = None
		printed_type = None
		captured_device = info[2]
		replayed_device = None
	capture = cv2.VideoCapture(video)
	n_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
	if capture.isOpened():
		pos_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
		fps = int(capture.get(cv2.CAP_PROP_FPS))
		while True:
			ret, frame = capture.read()
			if ret:
				name = str(uuid.uuid4()) + '.jpg'
				file_name = os.path.join(DIR_NAME, str(cls), name)
				frame = face_model.align(frame)
				if frame is None:
					continue
				cv2.imwrite(file_name, frame)
				result.append((name, cls, spoof_type, printed_type, captured_device, replayed_device))
			if pos_frame == n_frames:
				break
			if ret == False:
				break
			pos_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
	capture.release()

result = []
for root, dirs, files in os.walk(ROOT):
	if files:
		class_name = os.path.relpath(root, ROOT)
		if class_name == 'live':
			cls = 0
		else:
			cls = 1
		for file in files:
			print("Processing: {}".format(file))
			get_frames(os.path.join(root, file), file[:-4], result, cls)

print("Get {} images".format(len(result)))
data = pd.DataFrame(result, columns=['file_name', 'class', 'spoof_type', 'printed_type', 'captured_device', 'replayed_device'])
data.to_csv('data.csv', index=False)