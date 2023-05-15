import yaml
import math
from utils.predict_process.classifier import *
from utils.predict_process.detector import *
from utils.csv_process.save_csv_result import write_result_csv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class BadmintonAI(object):
    def __init__(self, config):
        self.path_config = config['PATH']
        self.weight_config = config['WEIGHTS']
        self.test_video = os.listdir(self.path_config['VIDEO'])
        self.video_path_list = []
        self.result_path = self.path_config['RESULT']
        self.detector = YoloDetector(self.weight_config['PLAYER'],
                                     self.weight_config['COURT'],
                                     self.weight_config['NET'])
        self.classifier = YoloClassifier(self.weight_config['ROUNDHEAD'], self.weight_config['BACKHAND'],
                                         self.weight_config['BALLTYPE'],
                                         self.weight_config['START'], self.weight_config['AFTER'])
        self.predict_result = []

        for folder in self.test_video:
            folder_path = os.path.join(self.path_config['VIDEO'], folder)
            mp4_files = glob.glob(os.path.join(folder_path, '*.mp4'))
            self.video_path_list.append(mp4_files)

    @staticmethod
    def _get_distance(point_1, point_2):
        x1, y1 = point_1
        x2, y2 = point_2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    @staticmethod
    def get_hit_frame_index_by_csv(video_path, csv_path):
        video_name = os.path.basename(video_path[0])
        hit_event_csv = os.path.join(csv_path, video_name[:-4] + '_predict.csv')
        print(f"==> Get the hitting frame from {hit_event_csv}")
        frame_index, image_opencv, ball_location= [], [], []
        with open(hit_event_csv, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                str_lines = lines[i].rstrip('\n') if '\n' in lines[i] else lines[i]
                frame_id, x, y = str_lines.split(',')
                frame_index.append(int(frame_id))
                ball_location.append([int(x), int(y)])

        cap = cv2.VideoCapture(video_path[0])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # frame number start from 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # frame number start from 0
            if frame_num in frame_index:
                image_opencv.append(frame)
        cap.release()

        return frame_index, image_opencv, ball_location

    def predict_each_image(self, video_name, frame_indexes, cv_image_list, ball_location):
        print(f"==> Predict {len(cv_image_list)} frame from video_{video_name}")
        previous_hitter = ''
        real_ShotSeq = 1
        for ShotSeq, cv_image in enumerate(tqdm(cv_image_list)):
            self.detector.set_image(cv_image, 'cpu')
            xy_A, xy_B = self.detector.player_detector.get_AB_player_position(center_point=True)
            if xy_A is None or xy_B is None: continue

            xy_ball = ball_location[ShotSeq]
            dist_A, dist_B = self._get_distance(xy_A, xy_ball), self._get_distance(xy_B, xy_ball)
            hitter = 'A' if dist_A < dist_B else 'B'
            if ShotSeq != 0 and previous_hitter == hitter: continue
            previous_hitter = hitter

            hitter_image, defender_image = self.detector.get_hitter_defender_image(hitter)
            os.makedirs(self.path_config['OPENPOSE'], exist_ok=True)
            video_stem = video_name[:-4]
            cv2.imwrite(os.path.join(self.path_config['OPENPOSE'],
                                     f'{video_stem}_frame_{frame_indexes[ShotSeq]}_hitter.png'), hitter_image)
            cv2.imwrite(os.path.join(self.path_config['OPENPOSE'], 
                                     f'{video_stem}_frame_{frame_indexes[ShotSeq]}_defender.png'), defender_image)

            self.classifier.set_image(hitter_image, 'cpu')
            RH_class = self.classifier.get_RH()
            BH_class = self.classifier.get_BH()
            ball_type = self.classifier.get_ball_type(separate=True, start=True if ShotSeq == 0 else False)

            ball_height = 2 # or 1

            winner = 'X' if ShotSeq != len(cv_image_list) - 1 else 'A' # or 'B'

            HitterLocationX = 640
            HitterLocationY = 360
            DefenderLocationX = 640
            DefenderLocationY = 360

            result = [video_name, real_ShotSeq, frame_indexes[ShotSeq], hitter, RH_class, BH_class, ball_height,
                      xy_ball[0], xy_ball[1], HitterLocationX, HitterLocationY, DefenderLocationX, DefenderLocationY,
                      ball_type, winner]
            self.predict_result.append(result)
            real_ShotSeq += 1


    def run_inference(self):
        for i, video in enumerate(self.video_path_list):
            print("==========================================")
            VideoName = os.path.basename(video[0])
            print(f"Start inference {VideoName}")
            frame_indexes, cv_images, ball_xy = self.get_hit_frame_index_by_csv(video, self.path_config['HIT_CSV'])

            self.predict_each_image(VideoName, frame_indexes, cv_images, ball_xy)

    def write_result(self):
        write_result_csv(self.result_path, self.predict_result)

if __name__ == "__main__":
    ## Load yaml configuration file
    with open('inference.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    badminton_core = BadmintonAI(opt)
    badminton_core.run_inference()
    badminton_core.write_result()










