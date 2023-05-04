from ultralytics import YOLO
import os
import glob
import numpy as np
from tqdm import tqdm
import cv2


class YoloDetector(object):
    def __init__(self, img=None, players_weight=None, court_weight=None, net_weight=None):
        self.input_data = cv2.imread(img)
        #self.input_data = image # cv2 image
        self.device = 'cpu'
        self.player_detector = PlayerDetector(self.input_data, players_weight, self.device)
        self.court_detector = CourtDetector(self.input_data, court_weight, self.device)
        self.net_detector = NetDetector(self.input_data, net_weight, self.device)
        self.output_image = None


    def get_hitter_image(self, player):
        A, B = self.player_detector.get_AB_player_image
        if player == 'A':
            return A
        elif player == 'B':
            return B


    def set_device(self, device):
        self.device = device

    def save_output_images(self):
        cv2.imwrite("./result.png", self.input_data)


class PlayerDetector(object):
    def __init__(self, image_cv, weight, device):
        self.image = image_cv
        self.detect_players = YOLO(weight) if weight else None
        self.result_player = self.detect_players.predict(source=self.image,
            save=False, imgsz=640, max_det=2, device=device
        ) if self.detect_players else None

    def get_AB_player_position(self):
        try:
            player1 = self.result_player[0][0]
            player2 = self.result_player[0][1]
        except:
            print('Our model only detect 1 player...')
            return [None, None]

        class_id = player1.boxes.cls.item()
        xyxy_1 = list(player1.boxes.xyxy.cpu().numpy()[0])
        xyxy_2 = list(player2.boxes.xyxy.cpu().numpy()[0])
        xyxy_1 = [int(i) for i in xyxy_1]
        xyxy_2 = [int(i) for i in xyxy_2]
        result = [xyxy_1, xyxy_2] if class_id == 1 else [xyxy_2, xyxy_1]
        return result  # [A, B] (far person, close person)

    def get_AB_player_image(self):  # return cv images
        xyxy_A, xyxy_B = self.get_AB_player_position()
        if xyxy_A is None or xyxy_B is None: return [None, None]

        img_player_A = self.image[xyxy_A[1]:xyxy_A[3], xyxy_A[0]:xyxy_A[2], :]
        img_player_B = self.image[xyxy_B[1]:xyxy_B[3], xyxy_B[0]:xyxy_B[2], :]

        return [img_player_A, img_player_B]

    def save_player_image(self, save_path='.'):
        players_image = self.image.copy()
        os.makedirs(save_path, exist_ok=True)
        xyxy_A, xyxy_B = self.get_AB_player_position()
        try:
            cv2.rectangle(players_image, (xyxy_A[0], xyxy_A[1]), (xyxy_A[2], xyxy_A[3]), (0, 0, 255), 3)
            cv2.rectangle(players_image, (xyxy_B[0], xyxy_B[1]), (xyxy_B[2], xyxy_B[3]), (0, 0, 255), 3)
            cv2.imwrite(os.path.join(save_path, f'players.png'), players_image)
        except:
            print('Our model only detect 1 player...')

    def save_crop_player_image(self, save_path='.'):
        os.makedirs(save_path, exist_ok=True)
        crop_A, crop_B = self.get_AB_player_image()
        try:
            cv2.imwrite(os.path.join(save_path, f'crop_player_A.png'), crop_A)
            cv2.imwrite(os.path.join(save_path, f'crop_player_B.png'), crop_B)
        except:
            print('Our model only detect 1 player...')

class CourtDetector(object):
    def __init__(self, image_cv, weight, device):
        self.image = image_cv
        self.detect_court = YOLO(weight) if weight else None
        self.result_court = self.detect_court.predict(source=self.image,
            save=False, imgsz=640, max_det=1, device=device
        ) if self.detect_court else None

    def get_court_region(self):
        court_region = self.result_court[0][0].masks.xy[0]
        court_region = [[int(i[0]), int(i[1])] for i in court_region]
        return court_region

    def save_court_image(self, save_path='.'):
        court_image = self.image.copy()
        os.makedirs(save_path, exist_ok=True)
        polygon_list = self.get_court_region()
        polygon_list = np.array(polygon_list, np.int32)
        polygon_list = polygon_list.reshape((-1, 1, 2))
        cv2.drawContours(court_image, [polygon_list], 0, (255, 0, 0), 3)
        cv2.imwrite(os.path.join(save_path, f'court.png'), court_image)


class NetDetector(object):
    def __init__(self, image_cv, weight, device):
        self.image = image_cv
        self.detect_net = YOLO(weight) if weight else None
        self.result_net = self.detect_net.predict(source=self.image,
            save=False, imgsz=640, max_det=1, boxes=True, device=device
        ) if self.detect_net else None

    def get_net_box(self):
        net_box = self.result_net[0][0]
        xyxy_net = list(net_box.boxes.xyxy.cpu().numpy()[0])
        xyxy_net = [int(i) for i in xyxy_net]
        return xyxy_net

    def save_net_image(self, save_path='.'):
        net_image = self.image.copy()
        os.makedirs(save_path, exist_ok=True)
        xyxy_net = self.get_net_box()
        cv2.rectangle(net_image, (xyxy_net[0], xyxy_net[1]), (xyxy_net[2], xyxy_net[3]), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(save_path, f'net.png'), net_image)

if __name__ == "__main__":
    test_img = r'D:\AICUP\Competition-2023-PyTorch-Badminton\datasets\gt_frames\video_0000_frame_000038.png'
    player = '../../trained_weights/yolov8s_players_detection.pt'
    court = '../../trained_weights/yolov8s-seg_court_detection.pt'
    net = '../../trained_weights/yolov8s-seg_net_detection.pt'

    #### Total ####
    V8Detector = YoloDetector(img=test_img, players_weight=player, court_weight=court, net_weight=net)
    V8Detector.set_device('0')
    V8Detector.save_output_images()

    #### Player detection ####
    xyxy_A, xyxy_B = V8Detector.player_detector.get_AB_player_position()
    img_A, img_B = V8Detector.player_detector.get_AB_player_image()
    V8Detector.player_detector.save_player_image(save_path='./players')
    V8Detector.player_detector.save_crop_player_image(save_path='./players_crop')

    #### Court detection ####
    region_point = V8Detector.court_detector.get_court_region()
    V8Detector.court_detector.save_court_image(save_path='./court')

    #### Court detection ####
    xyxy_net = V8Detector.net_detector.get_net_box()
    V8Detector.net_detector.save_net_image(save_path='./net')
