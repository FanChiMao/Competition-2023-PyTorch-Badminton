import cv2
import os
from tqdm import tqdm
import csv
from glob import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='Preprocess Training Data')
    parser.add_argument('--input_folder', default=r'C:\Jonathan\AICUP\dataset\Public\train', type=str)
    parser.add_argument('--save_folder', default=r'E:\dataset\gt_frames', type=str)
    args = parser.parse_args()

    inp_dir = args.input_folder
    tar_dir = args.save_folder
    os.makedirs(tar_dir, exist_ok=True)

    for i, folder in enumerate(tqdm(os.listdir(inp_dir))):
        folder_path = os.path.join(inp_dir, folder)
        mp4_files = glob(os.path.join(folder_path, '*.mp4'))
        csv_files = glob(os.path.join(folder_path, '*.csv'))


        with open(csv_files[0], newline='') as csvfile:
            rows = csv.reader(csvfile)
            essential_frame = []
            for row in rows:
                try:
                    # essential_frame.append(int(row[1]) - 10)
                    # essential_frame.append(int(row[1]) - 5)
                    essential_frame.append(int(row[1]))
                    # essential_frame.append(int(row[1]) + 5)
                    # essential_frame.append(int(row[1]) + 10)
                except:
                    continue

        cap = cv2.VideoCapture(mp4_files[0])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # frame number start from 0
        # essential_frame.append(total_frames - 40)
        # essential_frame.append(total_frames - 30)
        essential_frame.append(total_frames - 20)
        # essential_frame.append(total_frames - 10)
        # essential_frame.append(total_frames)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # frame number start from 0

            if frame_num in essential_frame:
                save_path = os.path.join(tar_dir, f"video_{i:04}_frame_{frame_num:06}.png")
                cv2.imwrite(save_path, frame)
        cap.release()




if __name__ == '__main__':
    main()

