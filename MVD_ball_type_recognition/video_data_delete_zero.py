# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:58:35 2023

@author: ms024
"""

import os
import cv2
import csv
import glob
import numpy as np

if __name__ == "__main__":
    train_dir = os.listdir(r"E:\Job\ASUS\AICUP\Public\train_test")
    csvfile_write = open(r'E:\Job\ASUS\AICUP\Public\train_del.csv', 'w', newline='')
    writer = csv.writer(csvfile_write)
    with open(r'E:\Job\ASUS\AICUP\Public\train.csv', newline='') as csvfile:
        rows = list(csv.reader(csvfile))
    
    print(f"before = {len(rows)}")
    
    cnt = 0
    cnnt = 0
    for train_file in train_dir:
        video_dir = os.path.join(r"E:\Job\ASUS\AICUP\Public\train_test", train_file)
        video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
        
        for vid_path in video_paths:
            cap = cv2.VideoCapture(vid_path)
            frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_cnt == 0:
                print(f"vid_path = {vid_path} ", end="")
                cnnt += 1
                idx = np.where(np.array(rows)[:, 0] == vid_path)
                if len(idx[0]) > 0:
                    print(f"idx = {idx[0][0]}", end="")
                    cnt += 1
                    del rows[idx[0][0]]
                print("")

    print(f"after = {len(rows)}, cnnt = {cnnt}, cnt = {cnt}")
    writer.writerows(rows)
    csvfile_write.close()