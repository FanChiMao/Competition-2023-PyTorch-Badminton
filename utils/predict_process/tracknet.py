import numpy as np
import cv2
import yaml
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
import os
from tqdm import tqdm
from glob import glob
from tensorflow.keras.models import *

BATCH_SIZE=1
HEIGHT=288
WIDTH=512
sigma=2.5
mag=1

def load_setting(yaml_path):
    with open(yaml_path, 'r') as config:
        opt = yaml.safe_load(config)
    PATH = opt['PATH']
    W = opt['WEIGHTS']
    return opt, PATH, W

def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
        return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap*mag


def custom_loss(y_true, y_pred):
    loss = (-1) * (K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (
                1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
    return K.mean(loss)


class TrackDetector(object):
    def __init__(self, track_weight=None, csv_path='./predict_csv'):
        self.track_detector = load_model(track_weight, custom_objects={'custom_loss':custom_loss}) if track_weight else None
        self.save_path = csv_path

    def	run_inference(self, video_path):
        """
        revise from TrackNetv2: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
        """
        cap = cv2.VideoCapture(video_path)

        success, image1 = cap.read()
        success, image2 = cap.read()
        success, image3 = cap.read()

        ratio = image1.shape[0] / HEIGHT
        os.makedirs(self.save_path, exist_ok=True)
        save_path = os.path.join(self.save_path, os.path.basename(video_path)[:-4] + '_predict.csv')

        f = open(save_path, 'w')
        f.write('Frame,Visibility,X,Y\n')

        while success:
            unit = []
            x1 = image1[...,::-1]
            x2 = image2[...,::-1]
            x3 = image3[...,::-1]
            x1 = array_to_img(x1)
            x2 = array_to_img(x2)
            x3 = array_to_img(x3)
            x1 = x1.resize(size = (WIDTH, HEIGHT))
            x2 = x2.resize(size = (WIDTH, HEIGHT))
            x3 = x3.resize(size = (WIDTH, HEIGHT))
            x1 = np.moveaxis(img_to_array(x1), -1, 0)
            x2 = np.moveaxis(img_to_array(x2), -1, 0)
            x3 = np.moveaxis(img_to_array(x3), -1, 0)
            unit.append(x1[0])
            unit.append(x1[1])
            unit.append(x1[2])
            unit.append(x2[0])
            unit.append(x2[1])
            unit.append(x2[2])
            unit.append(x3[0])
            unit.append(x3[1])
            unit.append(x3[2])
            unit=np.asarray(unit)
            unit = unit.reshape((1, 9, HEIGHT, WIDTH))
            unit = unit.astype('float32')
            unit /= 255
            y_pred = self.track_detector.predict(unit, batch_size=BATCH_SIZE)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype('float32')
            h_pred = y_pred[0]*255
            h_pred = h_pred.astype('uint8')

            if np.amax(h_pred) <= 0:
                f.write(str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)+',0,0,0\n')
            else:
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for i in range(len(rects)):
                    area = rects[i][2] * rects[i][3]
                    if area > max_area:
                        max_area_idx = i
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))

                f.write(str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)+',1,'+str(cx_pred)+','+str(cy_pred)+'\n')
            image1 = image2
            image2 = image3
            success, image3 = cap.read()
        f.close()




if __name__ == "__main__":
    """
    TrackNet is costing lots of time, you could consider run the predict csv first by this snippet.
    """

    OPT, P, W = load_setting('../../inference.yaml')

    TrackNet = TrackDetector(track_weight=r"../../trained_weights/tracknetv2_track_detection",
                             csv_path='./predict_csv')

    ## Predict hit frame by
    for i, folder in enumerate(tqdm(os.listdir(P['VIDEO']))):
        folder_path = os.path.join(P['VIDEO'], folder)
        mp4_files = glob(os.path.join(folder_path, '*.mp4'))

        TrackNet.run_inference(mp4_files[0])

