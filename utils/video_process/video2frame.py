import cv2
import os
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='Preprocess Training Data')
    parser.add_argument('--input_folder', default=r'D:\AICUP\Badminton\dataset\Public\train', type=str)
    parser.add_argument('--save_folder', default=r'D:\AICUP\Badminton\dataset\Public\train_frames', type=str)
    args = parser.parse_args()

    inp_dir = args.input_folder
    tar_dir = args.save_folder
    os.makedirs(tar_dir, exist_ok=True)

    for folder in tqdm(os.listdir(inp_dir)):
        folder_path = os.path.join(inp_dir, folder)
        mp4_files = glob(os.path.join(folder_path, '*.mp4'))
        for i, video in enumerate(mp4_files):
            cap = cv2.VideoCapture(video)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                save_path = os.path.join(tar_dir, f"v_{i:04}_frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):06}.png")
                cv2.imwrite(save_path, frame)
            cap.release()


def singlevideo2frame():
    path = r"D:\AICUP\Badminton\dataset\Public\train\00003\00003.mp4"
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('My Image', frame)
        cv2.waitKey(0)
    cap.release()



if __name__ == '__main__':
    # main()
    singlevideo2frame()
