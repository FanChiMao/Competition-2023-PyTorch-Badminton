import yaml
from glob import glob
import os
from tqdm import tqdm
import cv2

def load_setting(yaml_path):
    with open(yaml_path, 'r') as config:
        opt = yaml.safe_load(config)
    PATH = opt['PATH']
    W = opt['WEIGHTS']
    return opt, PATH, W















if __name__ == "__main__":
    ## Load yaml configuration file
    # OPT: all, P: paths, W: weights
    OPT, P, W = load_setting('inference.yaml')

    ## Predict hit frame by
    for i, folder in enumerate(tqdm(os.listdir(P['VIDEO']))):
        folder_path = os.path.join(P['VIDEO'], folder)
        mp4_files = glob(os.path.join(folder_path, '*.mp4'))

        cap = cv2.VideoCapture(mp4_files[0])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # frame number start from 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # frame number start from 0

        cap.release()

    pass









