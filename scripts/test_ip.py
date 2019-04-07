import cv2
import requests
import configs
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from video_stream import VideoStream
import os
import numpy as np
import eye_blink
import time
from PIL import Image, ImageStat
import datetime

PATH_TO_MODEL = './model'

def is_send_once_in_time(temp_now, prev_now):
    return temp_now - prev_now > configs.SEND_ONCE_IN_TIME

def is_valid_img(prev_now):
    temp_now = datetime.datetime.now().timestamp()
    return is_send_once_in_time(temp_now, prev_now)


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


def main(ip_cam_url=configs.IP_CAM_URL, username=configs.USERNAME, password=configs.PASSWORD):
    model = load_model()
    now_ = datetime.datetime.now().timestamp()
    r = requests.get(ip_cam_url, auth=(username, password), stream=True)
    if (r.status_code == 200):
        bytes_ = bytes()
        for chunk in r.iter_content(chunk_size=configs.CHUNK_SIZE):
            bytes_ += chunk
            a = bytes_.find(b'\xff\xd8')
            b = bytes_.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_[a:b + 2]
                bytes_ = bytes_[b + 2:]
                if is_valid_img(now_):
                    frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    face, bndbox = eye_blink.get_face(frame)
                    if face is None:
                        continue
                    try:
                        face = cv2.resize(face, (224, 224))
                    except:
                        continue
                    face = read_preprocess_image(face)  
                    j = model.predict(face)
                    #counter, total = eye_blink.detect_blink(frame, bndbox, counter, total)
                    draw_rec(frame, (0, 0, 255), bndbox) if j >= 1e-4 else draw_rec(frame, (0, 255, 0), bndbox)
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    print(j)
                    now_ = datetime.datetime.now().timestamp()
main()

                    
