import os
import glob
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import cv2

class YoloClassifier(object):    # Round Head    # Back Hand
    def __init__(self, RH_weight=None, BH_weight=None, balltype_weight=None):
        self.input_data = None
        self.device = 'cpu'
        self.weights = [RH_weight, BH_weight, balltype_weight]
        self.rh_classifier = None
        self.bh_classifier = None
        self.ball_types_classifier = None

    def set_image(self, cv_image=None, device='cpu'):
        self.input_data = cv_image
        self.set_device(device)
        self.rh_classifier = RoundHeadClassifier(cv_image, self.weights[0], self.device)
        self.bh_classifier = BackHandClassifier(cv_image, self.weights[1], self.device)
        self.ball_types_classifier = BallTypesClassifier(cv_image, self.weights[2], self.device)

    def set_device(self, device):
        self.device = device

    def get_RH(self):
        result_RH = self.rh_classifier.get_round_head_result()
        return result_RH

    def get_BH(self):
        result_BH = self.bh_classifier.get_back_hand_result()
        return result_BH

    def get_ball_type(self):
        ball_type = self.ball_types_classifier.get_ball_types_result()
        return ball_type


class RoundHeadClassifier(object):
    def __init__(self, image_cv, weight, device):
        self.image = image_cv
        self.classify_RH = YOLO(weight) if weight else None
        self.result_round_hand = self.classify_RH.predict(source=self.image,
            save=False, imgsz=224, device=device, verbose=False
        ) if self.classify_RH else None

    def get_round_head_result(self):
        class_dict = self.result_round_hand[0].names
        result_prob = self.result_round_hand[0].probs.cpu().numpy()
        predict_class = class_dict[np.argmax(result_prob)]
        return predict_class

class BackHandClassifier(object):
    def __init__(self, image_cv, weight, device):
        self.image = image_cv
        self.classify_BH = YOLO(weight) if weight else None
        self.result_back_hand = self.classify_BH.predict(source=self.image,
            save=False, imgsz=224, device=device, verbose=False
        ) if self.classify_BH else None

    def get_back_hand_result(self):
        class_dict = self.result_back_hand[0].names
        result_prob = self.result_back_hand[0].probs.cpu().numpy()
        predict_class = class_dict[np.argmax(result_prob)]
        return predict_class

class BallTypesClassifier(object):
    def __init__(self, image_cv, weight, device):
        self.image = image_cv
        self.classify_RH = YOLO(weight) if weight else None
        self.result_ball_types = self.classify_RH.predict(source=self.image,
            save=False, imgsz=224, device=device, verbose=False
        ) if self.classify_RH else None

    def get_ball_types_result(self):
        class_dict = self.result_ball_types[0].names
        result_prob = self.result_ball_types[0].probs.cpu().numpy()
        predict_class = class_dict[np.argmax(result_prob)]
        return predict_class

if __name__ == '__main__':
    image = cv2.imread(r'E:\AICUP\Badminton\all_crop_players\video_0000_000038_B.png')
    v8classifier = BallTypesClassifier(image, r'D:\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8n_cls_roundhead.pt', 'cpu')

    print(v8classifier.get_ball_types_result())