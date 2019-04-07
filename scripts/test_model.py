from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from video_stream import VideoStream
import cv2
import os
import numpy as np
import eye_blink
import time
from PIL import Image, ImageStat

PATH_TO_MODEL = './model'

def read_preprocess_image(img):
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def load_model():
    json_file = open(os.path.join(PATH_TO_MODEL, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(PATH_TO_MODEL, "first_try.h5"))
    print("Loaded model from disk")
    return loaded_model

def draw_rec(frame, color, bndbox):
	cv2.rectangle(frame, bndbox[0], bndbox[1], color, 2)

def get_brightness(img):
	img = Image.fromarray(img)
	stat = ImageStat.Stat(img)
	return stat.mean[0]

model = load_model()
counter = 0
total = 0
vs = VideoStream(src=0).start()
while True:
    frame = vs.read()
    face, bndbox = eye_blink.get_face(frame)
    if face is None:
        continue
    try:
        face = cv2.resize(face, (224, 224))
    except:
        continue
    #cv2.putText(frame, "Brightness: {}".format(get_brightness(face)), (300, 30),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    face = read_preprocess_image(face)  
    j = model.predict(face)
    counter, total = eye_blink.detect_blink(frame, bndbox, counter, total)
    draw_rec(frame, (0, 0, 255), bndbox) if j >= 1e-4 else draw_rec(frame, (0, 255, 0), bndbox)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        vs.stop()
        break
    print(j)
vs.stream.release()