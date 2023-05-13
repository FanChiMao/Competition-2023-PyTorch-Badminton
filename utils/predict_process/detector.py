from ultralytics import YOLO
import os
import glob
import numpy as np
from tqdm import tqdm
import cv2
from scipy.spatial import ConvexHull

class YoloDetector(object):
    def __init__(self, players_weight=None, court_weight=None, net_weight=None):
        self.input_data = None
        self.device = 'cpu'
        self.weights = [players_weight, court_weight, net_weight]
        self.player_detector = None
        self.court_detector = None
        self.net_detector = None

    def set_image(self, cv_image=None, device='cpu'):
        self.input_data = cv_image
        self.set_device(device)
        self.player_detector = PlayerDetector(cv_image, self.weights[0], self.device)
        self.court_detector = CourtDetector(cv_image, self.weights[1], self.device)
        self.net_detector = NetDetector(cv_image, self.weights[2], self.device)

    def set_device(self, device):
        self.device = device

    def get_hitter_image(self, player):
        A, B = self.player_detector.get_AB_player_image()
        if player == 'A':
            return A
        elif player == 'B':
            return B

    def save_output_images(self):
        cv2.imwrite("./result.png", self.input_data)


class PlayerDetector(object):
    def __init__(self, image_cv, weight, device):
        self.image = image_cv
        self.detect_players = YOLO(weight) if weight else None
        self.result_player = self.detect_players.predict(source=self.image,
            save=False, imgsz=640, max_det=2, device=device, verbose=False
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
            save=False, imgsz=640, max_det=1, device=device, verbose=False
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
            save=False, imgsz=640, max_det=1, boxes=True, device=device, verbose=False
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

def get_homography_matrix(TL, TR, ML, MR, BL, BR):
    from test_code.courtline import dst_coords
    from numpy.linalg import lstsq
    src_coords = np.hstack((TL, TR, ML, MR, BL, BR)).reshape(6, 2)
    src_coords_final = np.hstack((src_coords, np.ones((src_coords.shape[0], 1)))).astype("float32")
    M_transform = lstsq(src_coords_final[:, :], dst_coords[:, :], rcond=-1)[0]
    return M_transform

# def get_homography_matrix(ML, MR, Net_L, Net_R):
#     from numpy.linalg import lstsq
#     dst_coords = np.hstack((ML, MR)).reshape(2, 2)
#     src_coords = np.hstack((Net_L, Net_R)).reshape(2, 2)
#     src_coords_final = np.hstack((src_coords, np.ones((src_coords.shape[0], 1)))).astype("float32")
#     M_transform = lstsq(src_coords_final[:2, :], dst_coords[:2, :], rcond=-1)[0]
#     return M_transform

def mapping(src_coords, transform):
    tar_coords = np.matmul(np.array([[src_coords[0], src_coords[1], 1]]).astype("float32"), transform)
    return tar_coords

if __name__ == "__main__":
    test_img = r'D:\AICUP\datasets\gt_frames\video_0000_frame_000038.png'
    player = '../../trained_weights/yolov8s_players_detection_2.pt'
    court = '../../trained_weights/yolov8s-seg_court_detection.pt'
    net = '../../trained_weights/yolov8s-seg_net_detection.pt'

    img = cv2.imread(test_img)

    #### Total ####
    V8Detector = YoloDetector(img=test_img, players_weight=player, court_weight=court, net_weight=net)
    V8Detector.set_device('0')
    V8Detector.save_output_images()

    #### Player detection ####
    # xyxy_A, xyxy_B = V8Detector.player_detector.get_AB_player_position()
    # img_A, img_B = V8Detector.player_detector.get_AB_player_image()
    # V8Detector.player_detector.save_player_image(save_path='./players')
    # V8Detector.player_detector.save_crop_player_image(save_path='./players_crop')

    #### Court detection ####
    region_point = V8Detector.court_detector.get_court_region()
    min_x = min([point[0] for point in region_point])
    max_x = max([point[0] for point in region_point])
    min_y = min([point[1] for point in region_point])
    max_y = max([point[1] for point in region_point])

    top_x = []
    for point in region_point:
        if point[1] == min_y:
            top_x.append(point[0])

    TL = [min(top_x), min_y]
    TR = [max(top_x), min_y]

    BL = [min_x, max_y]
    BR = [max_x, max_y]

    print(TL)
    print(TR)
    print(BL)
    print(BR)
    #### Net detection ####
    xyxy_net = V8Detector.net_detector.get_net_box()
    ML, MR = [xyxy_net[0], xyxy_net[3]], [xyxy_net[2], xyxy_net[3]]
    Net_L, Net_R = [xyxy_net[0], xyxy_net[1]], [xyxy_net[2], xyxy_net[1]]


    #### Homography transformation ####
    # color = (0, 255, 0)
    # radius = 5
    # cv2.circle(img, BL, radius, color, -1)
    # cv2.circle(img, BR, radius, color, -1)
    # #cv2.circle(img, ML, radius, color, -1)
    # #cv2.circle(img, MR, radius, color, -1)
    # cv2.circle(img, TL, radius, color, -1)
    # cv2.circle(img, TR, radius, color, -1)
    # # cv2.circle(img, Net_L, radius, color, -1)
    # # cv2.circle(img, Net_R, radius, color, -1)
    # cv2.imwrite("corners.png", img)

    # homography = get_homography_matrix(TL, TR, ML, MR, BL, BR)
    # tar_coords = mapping(src_coords=[746,201], transform=homography)
    # x_t = tar_coords[0, 0]
    # y_t = tar_coords[0, 1]
    # print(tar_coords)

    # TOP_LEFT_2D = [42, 0]
    # TOP_RIGHT_2D = [568, 0]
    # MIDDLE_LEFT_2D = [0, 670]
    # MIDDLE_RIGHT_2D = [610, 670]
    # BOTTOM_LEFT_2D = [42, 1340]
    # BOTTOM_RIGHT_2D = [568, 1340]
    #
    # dst = np.array([BOTTOM_RIGHT_2D, BOTTOM_LEFT_2D, MIDDLE_RIGHT_2D, MIDDLE_LEFT_2D], np.float32)
    # src = np.array([BR, BL, MR, ML], np.float32)

    img_pts = np.float32([TL, TR, BL, BR])
    offset = 0
    # The coordinates of the four corners of the badminton court in the world
    world_pts = np.float32([[0 + offset, 0 + offset],
                            [544 + offset, 0 + offset],
                            [0 + offset, 1340 + offset],
                            [544 + offset, 1340 + offset]])

    # Compute the homography matrix
    H, _ = cv2.findHomography(img_pts, world_pts)

    # The coordinate of the shuttlecock in the image
    # Replace this with the actual coordinate from your image
    shuttlecock_img_pt = np.array([791,361, 1])

    # Use the homography matrix to transform the shuttlecock's image point to a world point
    shuttlecock_world_pt = np.dot(H, shuttlecock_img_pt)

    # Normalize the world point
    shuttlecock_world_pt /= shuttlecock_world_pt[2]

    print(shuttlecock_world_pt)
