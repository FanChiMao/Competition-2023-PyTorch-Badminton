import yaml
import glob
import os
from tqdm import tqdm
import cv2
from utils.predict_process.classifier import *
from utils.predict_process.detector import *
from utils.csv_process.save_csv_result import write_result_csv

class BadmintonAI(object):
    def __init__(self, config):
        self.path_config = config['PATH']
        self.weight_config = config['WEIGHTS']
        self.frame_selector = None
        self.test_video = os.listdir(self.path_config['VIDEO'])
        self.video_path_list = []
        self.predict_result = []
        self.result_path = self.path_config['RESULT']

        for folder in self.test_video:
            folder_path = os.path.join(self.path_config['VIDEO'], folder)
            mp4_files = glob.glob(os.path.join(folder_path, '*.mp4'))
            self.video_path_list.append(mp4_files)

    def get_key_frame_index(self, video_path):
        print("==> Get the hitting frame")
        frame_index = []
        image_opencv = []

        return frame_index, image_opencv

    def predict_each_image(self, video_name, frame_indexes, cv_image_list):
        print(f"==> Predict {len(cv_image_list)} frames from video_{video_name}")
        for ShotSeq, cv_image in enumerate(cv_image_list):
            V8Detector = YoloDetector(img=cv_image,
                         players_weight=self.weight_config['PLAYER'],
                         court_weight=self.weight_config['COURT'],
                         net_weight=self.weight_config['NET'])

            # TODO: Judge A or B player by the location of ball.
            hitter = 'A' or 'B'

            hitter_image = V8Detector.get_hitter_image(hitter)

            # TODO: classify roundhead and backhand
            V8Classifier = YoloClassifier(img=hitter_image, RH_weight=None, BH_weight=None, balltype_weight=None)
            RH_class = V8Classifier.get_RH()
            BH_class = V8Classifier.get_BH()

            # TODO: get ball location
            ball_x, ball_y = None, None

            # TODO: judge ball height
            ball_height = 1 or 2

            result = [video_name, str(ShotSeq + 1), frame_indexes[ShotSeq], hitter, RH_class, BH_class, ball_height,
                      ball_x, ball_y]

            self.predict_result.append(result)


    def run_inference(self):
        for i, video in enumerate(self.video_path_list):
            print("==========================================")
            VideoName = os.path.basename(video[0])
            print(f"Start inference {VideoName}")
            frame_indexes, cv_images = self.get_key_frame_index(video)

            self.predict_each_image(VideoName, frame_indexes, cv_images)

    def write_result(self):
        write_result_csv(self.result_path, self.predict_result)

if __name__ == "__main__":
    ## Load yaml configuration file
    with open('inference.yaml', 'r') as config:
        opt = yaml.safe_load(config)

    badminton_core = BadmintonAI(opt)
    badminton_core.run_inference()
    badminton_core.write_result()

    # cap = cv2.VideoCapture(mp4_files[0])
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # frame number start from 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # frame number start from 0
    #
    # cap.release()

    pass









