import os
import sys
import numpy as np
from scipy.signal import firwin 
import cv2
from mtcnn.mtcnn import MTCNN
from skin_color_filter import SkinColorFilter
from scipy.signal import filtfilt
from matplotlib import pyplot as plt

detector = MTCNN()
FRAMERATE = 61
ORDER = 128
THRESHOLD = 0.5
PLOT = False


def build_bandpass_filter(fs, order, min_freq=0.7, max_freq=4.0):
	min_freq = 0.7 
	max_freq = 4.0 

	nyq = fs / 2.0
	numtaps = order + 1
	b = firwin(numtaps, [min_freq/nyq, max_freq/nyq], pass_zero=False)
	return b

def scale_image(image, width, heigth):
	return cv2.resize(image, (width, heigth))

def crop_face(image, bbx, facewidth=None):
	if facewidth is None:
		facewidth = bbx[2]
	face = image[bbx[1]:(bbx[1] + bbx[3]), bbx[0]:(bbx[0] + bbx[2]), :]
	aspect_ratio = bbx[3] / bbx[2]
	faceheight = int(facewidth * aspect_ratio)
	face = scale_image(face, facewidth, faceheight)
	face = np.transpose(face, (2, 0, 1))
	return face

def compute_mean_rgb(image, mask=None):
	mean_r = np.mean(image[0, mask])
	mean_g = np.mean(image[1, mask])
	mean_b = np.mean(image[2, mask])
	return mean_r, mean_g, mean_b

def project_chrominance(r, g, b):
	x = (3.0 * r) - (2.0 * g)
	y = (1.5 * r) + g - (1.5 * b)
	return x, y

# def test(image):
# 	image = cv2.imread(image)
# 	faces = detector.detect_faces(image)
# 	bndbox = faces[0]['box']
# 	image = crop_face(image, bndbox)
# 	cv2.imshow('frame', image)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

#test(r'C:\Users\Assem\Desktop\face-spoofing\test1_3DDFA.jpg')

def main(video):
	cap = cv2.VideoCapture(video)
	nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	output_data = np.zeros(nb_frames, dtype='float64')
	chrom = np.zeros((nb_frames, 2), dtype='float64')
	fps = cap.get(cv2.CAP_PROP_FPS)
	bandpass_filter = build_bandpass_filter(fps, ORDER)
	if cap.isOpened():
		pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
		skin_filter = SkinColorFilter()		
		counter = 0
		while True:
			ret, frame = cap.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			if ret:
				faces = detector.detect_faces(frame)
				if (len(faces) > 0):
					bndbox = faces[0]['box']
					face = crop_face(frame, bndbox)
					print(face.shape)
					if counter == 0:
						skin_filter.estimate_gaussian_parameters(face)
					skin_mask = skin_filter.get_skin_mask(face, THRESHOLD)
					if np.count_nonzero(skin_mask) != 0:
						r,g,b = compute_mean_rgb(face, skin_mask)
						chrom[counter] = project_chrominance(r, g, b)
					else:
						print('No skin pixels detected')
						if counter == 0:
							chrom[counter] = project_chrominance(128., 128., 128.)
						else:
							chrom[counter] = chrom[counter - 1]
					counter += 1
				else:
					print('Cannot find face')
			if pos_frame == 100:
				break
			elif ret == False:
				break
			pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
	else:
		print('Some problems with video file: {}'.format(video))

	print(chrom)
	x_bandpassed = np.zeros(nb_frames, dtype='float64')
	y_bandpassed = np.zeros(nb_frames, dtype='float64')
	x_bandpassed = filtfilt(bandpass_filter, np.array([1]), chrom[:, 0], padlen=370)
	y_bandpassed = filtfilt(bandpass_filter, np.array([1]), chrom[:, 1], padlen=370)

	alpha = np.std(x_bandpassed) / np.std(y_bandpassed)
	pulse = x_bandpassed - alpha * y_bandpassed

	f, axarr = plt.subplots(1)
	#fft = np.fft.fft(pulse)
	plt.plot(range(pulse.shape[0]), pulse, 'k')
	plt.title("Pulse signal")
	plt.show()

if __name__ == "__main__":
	main('VID_20190304_144545.mp4')