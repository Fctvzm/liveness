import cv2
import dlib
import numpy as np
from collections import OrderedDict

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

class Face:

	def __init__(self, 
		face_predictor_model='shape_predictor_68_face_landmarks.dat', 
		face_recognition_model=None, 
		desired_left_eye=(0.4, 0.4), 
		desired_face_width=224, 
		desired_face_heigth=None):

		if (face_recognition_model is None):
			print('model to encode is not loaded')
			self.encoder = None
		else:
			self.encoder = dlib.face_recognition_model_v1(face_recognition_model) #model outputs 128d vector / compute face desccriptor

		if (face_predictor_model is None):
			print('model to predict is not loaded')
			self.predictor = None
		else:
			self.predictor = dlib.shape_predictor(face_predictor_model) #find locations of landmarks / pretrained model

		self.detector = dlib.get_frontal_face_detector() #find faces in the image / uses hog 
		self.desired_left_eye = desired_left_eye #specifying the desired output left eye position
		self.desired_face_width = desired_face_width #defines desired face width in pixels
		self.desired_face_heigth = desired_face_heigth #desired face height value in pixels

		if self.desired_face_heigth is None: #image wil be square
			self.desired_face_heigth = self.desired_face_width

	def rect_to_bndbox(self, rect):
		#convert rectagnle to bounding box representation
		return rect.top(), rect.right(), rect.bottom(), rect.left()

	def bndbox_to_rect(self, bndbox):
		#convert bounding box tuple to rectangle format
		return dlib.rectangle(bndbox[3], bndbox[0], bndbox[1], bndbox[2])

	def check_bounds(self, bndbox, image_shape):
		#verify that bounding box within the bounds of image
		return max(bndbox[0], 0), min(bndbox[1], image_shape[1]), min(bndbox[2], image_shape[0]), max(bndbox[3], 0)

	def face_locations(self, img, n_upsample=1):
		#detect faces in image
		return [self.check_bounds(self.rect_to_bndbox(face), img.shape) for face in self.detector(img, n_upsample)]

	def face_landmarks(self, img, face_locations=None):
		#detect the landmarks in face
		assert self.predictor is not None

		if face_locations is None:
			face_locations = self.detector(img)
		else:
			face_locations = [self.bndbox_to_rect(location) for location in face_locations]

		return [self.predictor(img, location) for location in face_locations]

	def landmarks_to_tuple(self, landmarks):
		#return landmarks as x, y coordinates
		return [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]
		#return np.asarray([(p.x, p.y) for p in landmarks[0].parts()])

	# def face_encodings(self, img, face_locations=None, n_sample=1):
	# 	#return 128d vector based on found landmark points
	# 	assert self.encoder is not None

	# 	landmarks = self.face_landmarks(img, face_locations)
	# 	return [np.array(self.encoder.compute_face_descriptor(img, landmarks_set, n_sample)) for landmarks_set in landmarks]

	# def compare_faces(self, face_encodings, face_to_compare, threshold=0.6):
	# 	#find eucladian distance and threshold it
	# 	return np.linalg.norm(face_encodings - face_to_compare, axis=1)

	def get_center_dist(self, bndbox, center_point):
		x1 = int((bndbox[1] + bndbox[3]) / 2)
		y1 = int((bndbox[0] + bndbox[2]) / 2)
		x2, y2 = center_point
		return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

	def get_max_bndbox(self, face_locations, center_point):
		center_dists = [self.get_center_dist(face_location, center_point) for face_location in face_locations]
		return np.argmin(center_dists)

	def align(self, img):
		#aling image with eyes in one x horizontal line
		center_x = int(img.shape[1] / 2)
		center_y = int(img.shape[0] / 2)
		bndboxes = self.face_locations(img)
		if (len(bndboxes) == 0):
			print('could not find face in image')
			return None
		i = self.get_max_bndbox(bndboxes, (center_x, center_y))
		locations = []
		locations.append(bndboxes[i])
		landmarks = self.face_landmarks(img, locations)
		landmarks_tuple = np.squeeze(self.landmarks_to_tuple(landmarks))

		#extract the left and right eye (x, y)-coordinates
		(l_start, l_end) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
		(r_start, r_end) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
		left_eye = landmarks_tuple[l_start:l_end]
		right_eye = landmarks_tuple[r_start:r_end]

		#compute the center of mass for each eye
		left_eye_center = left_eye.mean(axis=0).astype("int")
		right_eye_center = right_eye.mean(axis=0).astype("int")

		#compute the angle between the eye centroids
		dY = right_eye_center[1] - left_eye_center[1]
		dX = right_eye_center[0] - left_eye_center[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		#compute the desired right eye x-coordinate
		desired_right_eye_x = 1.0 - self.desired_left_eye[0]

		#determine the scale of the new resulting image by taking
		#the ratio of the distance between eyes in the current
		#image to the ratio of distance between eyes in the
		#desired image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
		desired_dist *= self.desired_face_width
		scale = desired_dist / dist

		#compute center (x, y)-coordinates between the two eyes in the input image
		eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
			(left_eye_center[1] + right_eye_center[1]) // 2)

		#rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

		#update the translation component of the matrix
		tX = self.desired_face_width * 0.5
		tY = self.desired_face_heigth * self.desired_left_eye[1]
		M[0, 2] += (tX - eyes_center[0])
		M[1, 2] += (tY - eyes_center[1])

		#apply the affine transformation
		(w, h) = (self.desired_face_width, self.desired_face_heigth)
		output = cv2.warpAffine(img, M, (w, h),
			flags=cv2.INTER_CUBIC)

		return output