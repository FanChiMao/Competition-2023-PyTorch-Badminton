import os
import glob
import numpy as np
from tqdm import tqdm
import cv2

class YoloClassifier(object):    # Round Head    # Back Hand
    def __init__(self, img=None, RH_weight=None, BH_weight=None, balltype_weight=None):
        self.input_data = img # cv2 image
        self.device = 'cpu'
        self.round_hand = None
        self.back_hand = None
        self.net_detector = None
        self.output_image = None


    def get_RH(self):
        return self.round_hand

    def get_BH(self):
        return self.round_hand