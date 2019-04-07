    import numpy as np
import cv2
from skimage.feature import local_binary_pattern,multiblock_lbp
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
from sklearn.svm import LinearSVC,SVC
from xgboost import XGBClassifier
import time

import argparse

# img = cv2.imread('madi.jpg')
# img = cv2.resize(img, (128, 128))
# imgYCC = cv2.cv2tColor(img, cv2.COLOR_BGR2YCR_CB)


PATH = '/home/avsoft/getframes/output/Train/'
# face_cascade = cv2.CascadeClassifier('/home/vakidzaci/cv/.env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def gethist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    radius = 2
    n_points = 16
    l = local_binary_pattern(gray, n_points, radius, method='uniform')
    histogram = l.ravel()
#    features = (histogram - np.mean(histogram)) / np.std(histogram)
    return histogram

ap = argparse.ArgumentParser()
ap.add_argument("-r","--rows",required=False, help="Number of test rows")
# ap.add_argument("-n", "--name", required=True, help="name of trained model to perform spoofing detection")
# ap.add_argument("-d", "--device", required=True, help="camera identifier/video to acquire the image")
# ap.add_argument("-t", "--threshold", required=False, help="threshold used for the classifier to decide between genuine and a spoof attack")
args = vars(ap.parse_args())

if __name__ == "__main__":
    r = args["rows"]
    train = pd.read_csv('/home/avsoft/getframes/output/train.csv')
    test = pd.read_csv('/home/avsoft/getframes/output/test.csv')
    if r is not None:
        r = int(r)
        train = train.head(r)
#        test = client_train.head(r)



    H = np.array([])
    y_train = []
    for index, row in train.iterrows():
        if row['class'] == 0:
            stype = 'live'
        else:
            stype = 'spoof'
        img = cv2.imread(PATH + stype + '/' + row['file_name'])
        if img is not None:
            hist = gethist(img)
            if(H.shape[0] == 0):
                H = np.array([hist])
            else:
                H = np.append(H,[np.array(hist)],axis=0)
            y_train.append(row['class'])


    y_train = np.asarray(y_train)



    print(H.shape)
    print(y_train.shape)

    clf = XGBClassifier()
    clf.fit(H,y_train)
    dump(clf, '/home/avsoft/color_texture_analysis/models/model0.joblib')
