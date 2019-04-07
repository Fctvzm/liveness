import dlib
from collections import OrderedDict
import cv2
import os
import numpy as np
from scipy.spatial import distance as dist

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

PATH_FACE_DETECTOR = './face_detector'
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
proto_path = os.path.join(PATH_FACE_DETECTOR, 'deploy.prototxt.txt')
model_path = os.path.join(PATH_FACE_DETECTOR, 'res10_300x300_ssd_iter_140000.caffemodel')
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)
	return ear

def shape_to_np(shape, dtype='int'):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

def get_face(frame):
	h, w = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			startX, startY, endX, endY = box.astype("int")
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		else:
			return None, None
	return face, ((startX, startY), (endX, endY))

def detect_blink(frame, bndbox, counter, total):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rec = dlib.rectangle(bndbox[0][0], bndbox[0][1], bndbox[1][0], bndbox[1][1])
	shape = predictor(gray, rec)
	shape = shape_to_np(shape)

	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]
	leftEAR = eye_aspect_ratio(leftEye)
	rightEAR = eye_aspect_ratio(rightEye)

	ear = (leftEAR + rightEAR) / 2.0

	leftEyeHull = cv2.convexHull(leftEye)
	rightEyeHull = cv2.convexHull(rightEye)
	cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

	if ear < EYE_AR_THRESH:
		counter += 1

	else:
		if counter >= EYE_AR_CONSEC_FRAMES:
			total += 1

		counter = 0

	cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
	cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

	return counter, total

