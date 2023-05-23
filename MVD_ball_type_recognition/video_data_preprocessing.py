# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 21:52:24 2023

@author: ms024
"""

import os
import cv2
import csv
import glob
from ultralytics import YOLO

def pad_and_resize(img):
    h, w, _ = img.shape
    max_size = max(h, w)
    if (max_size - h) > 0:
        top = bottom = (max_size - h)//2
    else:
        top = bottom = 0
    
    if (max_size - w) > 0:
        left = right = (max_size - w)//2
    else:
        left = right = 0
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return img

def get_csv_data(csv_path):
    with open(csv_path, newline='') as csvfile:
        rows = list(csv.reader(csvfile))
    hit_frame_idx = rows[0].index('HitFrame')
    hitter_idx = rows[0].index('Hitter')
    ball_type_idx = rows[0].index('BallType')
    
    csv_data = []
    for i in range(1, len(rows)):
        csv_data.append([int(rows[i][hit_frame_idx]), rows[i][hitter_idx], int(rows[i][ball_type_idx])])
    return csv_data ### HitFrame, Hitter, BallType

if __name__ == "__main__":
    train_dir = os.listdir(r"E:\Job\ASUS\AICUP\Public\train")
    csvfile_write = open(r'E:\Job\ASUS\AICUP\Public\train.csv', 'w', newline='')
    writer = csv.writer(csvfile_write)
    
    detect_players = YOLO(r"E:\Job\ASUS\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8s_players_detection.pt")
    
    for train_file in train_dir:
        video_dir = os.path.join(r"E:\Job\ASUS\AICUP\Public\train", train_file)
        csv_path = glob.glob(os.path.join(video_dir, "*.csv"))
        video_path = glob.glob(os.path.join(video_dir, "*.mp4"))
    
        csv_data = get_csv_data(csv_path[0])
        cap = cv2.VideoCapture(video_path[0])
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        try:
            os.mkdir(video_dir.replace("train", "train_test"))
        except Exception as e:
            print(e)
        
        out = cv2.VideoWriter(video_path[0].replace("train", "train_test").split(".mp4")[0]+"_0.mp4", fourcc, fps, (224,  224))  # 產生空的影片
        out_none = cv2.VideoWriter(video_path[0].replace("train", "train_test").split(".mp4")[0]+"_none_0.mp4", fourcc, fps, (224,  224))

        frame_cnt = 1
        clip_idx = 0
        key_frame_range = 8
    
        while(True):
            ret, frame = cap.read()
            if not ret:
                break
            
            print(f"frame {frame_cnt}", end="")
            result_player = detect_players.predict(source=frame, save=False, imgsz=640, max_det=2, device='cpu')
            try:
                player1 = result_player[0][0]
                player2 = result_player[0][1]
            except:
                print('Our model only detect 1 player...')
    
            class_id = player1.boxes.cls.item()
            xyxy_1 = list(player1.boxes.xyxy.cpu().numpy()[0])
            xyxy_2 = list(player2.boxes.xyxy.cpu().numpy()[0])
            xyxy_1 = [int(i) for i in xyxy_1]
            xyxy_2 = [int(i) for i in xyxy_2]
            xyxy_A, xyxy_B = [xyxy_1, xyxy_2] if class_id == 1 else [xyxy_2, xyxy_1]
            
            if xyxy_A[1] >= xyxy_B[1]: ### upper A, bottom B
                img_player_B = frame[xyxy_A[1]:xyxy_A[3], xyxy_A[0]:xyxy_A[2], :]
                img_player_A = frame[xyxy_B[1]:xyxy_B[3], xyxy_B[0]:xyxy_B[2], :]
                
            else:
                img_player_A = frame[xyxy_A[1]:xyxy_A[3], xyxy_A[0]:xyxy_A[2], :]
                img_player_B = frame[xyxy_B[1]:xyxy_B[3], xyxy_B[0]:xyxy_B[2], :]
            
            img_player_A = pad_and_resize(img_player_A)
            img_player_B = pad_and_resize(img_player_B)
            
            # cv2.imshow('img_player_A', img_player_A)
            # cv2.imshow('img_player_B', img_player_B)
            
            frame_cnt+=1
            if clip_idx < len(csv_data):
                key_frame = csv_data[clip_idx][0]
                hitter_frame = csv_data[clip_idx][1]
                if frame_cnt < key_frame - (2*key_frame_range):
                    if hitter_frame == "A":
                        out_none.write(img_player_A)
                    elif hitter_frame == "B":
                        out_none.write(img_player_B)
                elif frame_cnt == key_frame - (2*key_frame_range) and (clip_idx < len(csv_data)-1):
                    
                    writer.writerow([video_path[0].replace("train", "train_test").split(".mp4")[0]+f"_none_{clip_idx}.mp4", 0])
                    out_none.release()
                    out_none = cv2.VideoWriter(video_path[0].replace("train", "train_test").split(".mp4")[0]+f"_none_{clip_idx}.mp4", fourcc, fps, (224,  224))
                
                if (frame_cnt >= key_frame - key_frame_range) and (frame_cnt < key_frame + key_frame_range):
                    print(f", hitter = {hitter_frame}, ball type = {csv_data[clip_idx][2]} ", end="")
                    if hitter_frame == "A":
                        out.write(img_player_A)
                    elif hitter_frame == "B":
                        out.write(img_player_B)
                        
                elif frame_cnt == key_frame + key_frame_range:
                    writer.writerow([video_path[0].replace("train", "train_test").split(".mp4")[0]+f"_{clip_idx}.mp4", csv_data[clip_idx][2]])
                    clip_idx += 1
                    out.release()
                    if clip_idx < len(csv_data):
                        
                        out = cv2.VideoWriter(video_path[0].replace("train", "train_test").split(".mp4")[0]+f"_{clip_idx}.mp4", fourcc, fps, (224,  224))
            
            # cv2.imshow('frame', frame)
            print("")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        out_none.release()
    csvfile_write.close()