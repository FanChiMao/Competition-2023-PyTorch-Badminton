import cv2
import os
from tqdm import tqdm
import csv
from glob import glob
import argparse
from utils.predict_process.detector import YoloDetector, PlayerDetector


def build_sub_folder_path(save_path):
    round_head = os.path.join(save_path, 'round_head')
    back_hand = os.path.join(save_path, 'back_hand')
    ball_type = os.path.join(save_path, 'ball_type')
    os.makedirs(round_head, exist_ok=True)
    os.makedirs(back_hand, exist_ok=True)
    os.makedirs(ball_type, exist_ok=True)

    os.makedirs(os.path.join(round_head, '1'), exist_ok=True)
    os.makedirs(os.path.join(round_head, '2'), exist_ok=True)
    os.makedirs(os.path.join(back_hand, '1'), exist_ok=True)
    os.makedirs(os.path.join(back_hand, '2'), exist_ok=True)

    ball_type_all = os.path.join(ball_type, 'all')
    ball_type_separate = os.path.join(ball_type, 'separate')
    ball_type_separate_start = os.path.join(ball_type_separate, 'start')
    ball_type_separate_after = os.path.join(ball_type_separate, 'after')

    for i in range(1, 10):
        os.makedirs(os.path.join(ball_type_all, str(i)), exist_ok=True)
        if i in [1, 2]:
            os.makedirs(os.path.join(ball_type_separate_start, str(i)), exist_ok=True)
        else:
            os.makedirs(os.path.join(ball_type_separate_after, str(i)), exist_ok=True)

    return round_head, back_hand, ball_type_all, ball_type_separate_start, ball_type_separate_after


def main():
    parser = argparse.ArgumentParser(description='Preprocess Training Data')
    parser.add_argument('--input_folder', default=r'C:\Jonathan\AICUP\dataset\Public\train', type=str)
    parser.add_argument('--save_folder', default=r'C:\Jonathan\AICUP\dataset\players', type=str)
    parser.add_argument('--player_weight', default=r'C:\Jonathan\AICUP\Competition-2023-PyTorch-Badminton'
                                                   r'\trained_weights\yolov8s_players_detection.pt', type=str)
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
            player_index = []
            round_head = []
            back_hand = []
            ball_type = []  # 1~9
            for row in rows:
                try:
                    essential_frame.append(int(row[1]))
                    player_index.append(row[2])
                    round_head.append(row[3])
                    back_hand.append(row[4])
                    ball_type.append(row[-2])
                except:
                    continue

        assert len(essential_frame) == len(player_index) == len(round_head) == len(back_hand) == len(ball_type)

        round_head_path, back_hand_path, ball_type_all_path, \
            ball_type_separate_start_path, ball_type_separate_after_path = build_sub_folder_path(tar_dir)

        cap = cv2.VideoCapture(mp4_files[0])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # frame number start from 0

            if frame_num in essential_frame:
                save_frame_name = f"video_{i:04}_{frame_num:06}_hitter.png"
                index = essential_frame.index(frame_num)
                img_player_A, img_player_B = PlayerDetector(frame, args.player_weight, device=0).get_AB_player_image()
                if img_player_A is None or img_player_B is None: continue
                crop_image = img_player_A.copy() if player_index[index] == 'A' else img_player_B.copy()

                save_roundhead = os.path.join(os.path.join(round_head_path, round_head[index]), save_frame_name)
                cv2.imwrite(save_roundhead, crop_image)

                save_backhand = os.path.join(os.path.join(back_hand_path, back_hand[index]), save_frame_name)
                cv2.imwrite(save_backhand, crop_image)

                if ball_type[index] in ['1', '2']:
                    save_ball_type_start = os.path.join(
                        os.path.join(ball_type_separate_start_path, ball_type[index]), save_frame_name)
                    cv2.imwrite(save_ball_type_start, crop_image)
                else:
                    save_ball_type_after = os.path.join(
                        os.path.join(ball_type_separate_after_path, ball_type[index]), save_frame_name)
                    cv2.imwrite(save_ball_type_after, crop_image)

                save_ball_type_all = os.path.join(os.path.join(ball_type_all_path, ball_type[index]), save_frame_name)
                cv2.imwrite(save_ball_type_all, crop_image)

        cap.release()


if __name__ == '__main__':
    main()
