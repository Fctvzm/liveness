import uuid
import cv2
import numpy as np 
import pandas as pd
import os
from scipy.ndimage import uniform_filter
import matplotlib
from skimage import feature

CLASS = 0
TYPE = 1
PRINTED_TYPE = None
REPLAYED = None
ROOT = "sfsfs"


# def hog_feature(im):
# 	fd, hog_image = feature.hog(im, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True, feature_vector=True)
# 	return fd

# def color_histogram_hsv(image, nbin=10, xmin=0, xmax=255, normalized=True):
#   ndim = image.ndim
#   bins = np.linspace(xmin, xmax, nbin+1)
#   hsv = matplotlib.colors.rgb_to_hsv(image/xmax) * xmax
#   imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
#   imhist = imhist * np.diff(bin_edges)

#   return imhist

def get_info(file_name):
	return file_name.split('_')

def get_blur_value(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def get_difference(gray, image, hist, color_hist):
	return np.sum(abs(hog_feature(gray) - hist)), np.sum(abs(color_histogram_hsv(image) - color_hist))

def get_frames(video, video_name, result):
	main_hist = 0
	main_color_hist = 0
	info = get_info(video_name)
	subject_name = info[1]
	captured_device = info[2]
	capture = cv2.VideoCapture(video)
	n_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
	if capture.isOpened():
		pos_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
		fps = int(capture.get(cv2.CAP_PROP_FPS))
		while True:
			ret, frame = capture.read()
			if ret:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				blur = get_blur_value(gray)
				if (blur >= 300):
					name = str(uuid.uuid4()) + '.jpg'
					file_name = 'C:/Users/Assem/Desktop/face-spoofing/live-photos-exper/' + name
					cv2.imwrite(file_name, frame)
					result.append((name, TYPE))
			if pos_frame == n_frames:
				break
			if ret == False:
				break
			pos_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
	capture.release()


result = []
for root, dirs, files in os.walk(r'C:\Users\Assem\Desktop\live-faces'):
	if files:
		for file in files:
			print("Processing: {}".format(file))
			get_frames(os.path.join(root, file), file[:-4], result)

print("Get {} images".format(len(result)))
data = pd.DataFrame(result, columns=['file_name', 'class'])
data.to_csv('data.csv', index=False)