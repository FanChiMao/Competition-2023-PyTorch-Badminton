import os
import cv2
import csv
import json
import shutil
import numpy as np
import argparse


def MakeLabelFromRaw(file_path, n_frames, task_list):

    hit_frame_list = []
    hitter_list = []
    # 讀取csv檔案
    with open(file_path, "r") as csv_file:
        csv_data = csv.reader(csv_file)
        header = next(csv_data)
        # 取得指定欄位的索引
        hit_frame_row_index = header.index('HitFrame')
        hitter_row_index = header.index('Hitter')

        # 依序讀取每一列數據，並取得指定欄位的數值
        for row in csv_data:
            hit_frame_list.append(row[hit_frame_row_index])
            hitter_list.append(row[hitter_row_index])

    # make hit frame label
    hitframe_label = np.array([0]*n_frames)
    if 'HITFRAME_PRED' in task_list:
        for frame_idx in hit_frame_list:
            if int(frame_idx) == 0:
                print("zero index")

            true_index = int(frame_idx)
            if true_index-2 < 0 or true_index+2 >= len(hitframe_label):
                print("Boundary Error")
                print("HitFrame index: ", true_index)

            forward_2 = len(hitframe_label)-1 if true_index+2 >= len(hitframe_label) else true_index+2
            forward_1 = len(hitframe_label)-1 if true_index+1 >= len(hitframe_label) else true_index+1
            backward_2 = 0 if true_index-2 < 0 else true_index-2
            backward_1 = 0 if true_index-1 < 0 else true_index-1
            hitframe_label[backward_2] = 1
            hitframe_label[backward_1] = 1
            hitframe_label[true_index] = 1
            hitframe_label[forward_1] = 1
            hitframe_label[forward_2] = 1

    # make direction label
    direction_label = np.array([0]*n_frames)
    if 'DIRECTION_PRED' in task_list:
        for i in range(len(hit_frame_list)-1):
            from_idx = int(hit_frame_list[i])
            to_idx = int(hit_frame_list[i+1])
            
            if hitter_list[i] == 'A':
                direction_label[from_idx:to_idx] = 1
            else:
                direction_label[from_idx:to_idx] = 2

    hit_count = len(hit_frame_list)
    
    return hitframe_label.tolist(), direction_label.tolist(), hit_count

def main():
    parser = argparse.ArgumentParser()

    # preprocess raw args
    parser.add_argument('--raw-dir', type=str, default='D:/Dataset/AICUP/part1/train')
    parser.add_argument('--video-dir', type=str, default='../custom_data/videos/badminton_clean')
    parser.add_argument('--label-dir', type=str, default='../custom_data/labels/badminton_clean')
    parser.add_argument('--dirty-list', type=str, default='D:/Dataset/AICUP/part1/dirty_data.csv')
    parser.add_argument('--only-clean', type=bool, default=True)
    parser.add_argument('--tasks', type=str, nargs='+', default=['HITFRAME_PRED', 'DIRECTION_PRED'])


    args = parser.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.label_dir, exist_ok=True)

    dirty_list = []
    if args.only_clean:
        # 讀取csv檔案
        with open(args.dirty_list, "r") as csv_file:
            csv_data = csv.reader(csv_file)
            for row in csv_data:
                dirty_list.append(int(row[0])-1)
    
    # 列出目標資料夾下的所有子資料夾
    sub_folders = os.listdir(args.raw_dir)

    # 依序存取每個子資料夾中的csv和mp4檔案
    for i, sub_folder in enumerate(sub_folders):
        if i in dirty_list:
            continue

        sub_folder_path = os.path.join(args.raw_dir, sub_folder)
        csv_file_path = os.path.join(sub_folder_path, sub_folder + "_S2.csv")
        mp4_file_path = os.path.join(sub_folder_path, sub_folder + ".mp4")
        
        # 讀取mp4檔案
        cap = cv2.VideoCapture(mp4_file_path)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            
        # 釋放影像物件
        cap.release()

        hf_label, dr_label, _ = MakeLabelFromRaw(csv_file_path, frame_idx, args.tasks)

        labels = [hf_label, dr_label]
        json_label = {
            'user_summary': labels
        }

        # save labels.json to target folder
        json_path = os.path.join(args.label_dir, sub_folder + ".json")
        with open(json_path, 'w') as f:
            json.dump(json_label, f)
        
        # copy videos to target folder
        output_mp4_path = os.path.join(args.video_dir, sub_folder + ".mp4")
        shutil.copy(mp4_file_path, output_mp4_path)


if __name__ == '__main__':
    main()
