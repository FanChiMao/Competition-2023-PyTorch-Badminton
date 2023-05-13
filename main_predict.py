import yaml
from tqdm import tqdm
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
        self.classifier = YoloClassifier(self.weight_config['ROUNDHEAD'],
                                         self.weight_config['BACKHAND'],
                                         self.weight_config['BALLTYPE'])
        self.predict_result = []

        for folder in self.test_video:
            folder_path = os.path.join(self.path_config['VIDEO'], folder)
            mp4_files = glob.glob(os.path.join(folder_path, '*.mp4'))
            self.video_path_list.append(mp4_files)

    @staticmethod
    def get_hit_frame_index(video_path):
        print("==> Get the hitting frame")
        frame_index = []
        image_opencv = []
        return frame_index, image_opencv

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

    def predict_each_image(self, video_name, frame_indexes, cv_image_list, ball_location, last):
        print(f"==> Predict {len(cv_image_list)} frame from video_{video_name}")

        for ShotSeq, cv_image in enumerate(tqdm(cv_image_list)):
            self.detector.set_image(cv_image, 'cpu')

            # TODO: Judge A or B player by the location of ball.
            hitter = 'A' # or 'B'
            hitter_image = self.detector.get_hitter_image(hitter)


            # TODO: classify roundhead and backhand
            self.classifier.set_image(hitter_image, 'cpu')
            RH_class = self.classifier.get_RH()
            BH_class = self.classifier.get_BH()
            ball_type = self.classifier.get_ball_type()

            # TODO: get ball location
            ball_x, ball_y = ball_location[ShotSeq]

            # TODO: judge ball height
            ball_height = 2 # or 1

            # TODO: judge WINNER
            winner = 'X' if not last else 'A' # or 'B'

            HitterLocationX = 640
            HitterLocationY = 360
            DefenderLocationX = 640
            DefenderLocationY = 360

            result = [video_name, str(ShotSeq + 1), frame_indexes[ShotSeq], hitter, RH_class, BH_class, ball_height,
                      ball_x, ball_y, HitterLocationX, HitterLocationY, DefenderLocationX, DefenderLocationY, ball_type,
                      winner]

            self.predict_result.append(result)


    def run_inference(self):
        for i, video in enumerate(self.video_path_list):
            last = True if i == len(self.video_path_list)-1 else False
            print("==========================================")
            VideoName = os.path.basename(video[0])
            print(f"Start inference {VideoName}")
            # frame_indexes, cv_images = self.get_hit_frame_index(video)
            frame_indexes, cv_images, ball_xy = self.get_hit_frame_index_by_csv(video, self.path_config['HIT_CSV'])

            self.predict_each_image(VideoName, frame_indexes, cv_images, ball_xy, last)

    def write_result(self):
        write_result_csv(self.result_path, self.predict_result)

if __name__ == "__main__":
    ## Load yaml configuration file
    with open('inference.yaml', 'r') as config:
        opt = yaml.safe_load(config)

    badminton_core = BadmintonAI(opt)
    badminton_core.run_inference()


    badminton_core.write_result()


    pass









