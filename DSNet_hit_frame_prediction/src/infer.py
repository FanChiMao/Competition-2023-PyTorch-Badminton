import numpy as np
import logging
import os
import torch
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

from helpers import init_helper, video_helper
from modules.model_zoo import get_model


logger = logging.getLogger()

def main():
    args = init_helper.get_arguments()

    # Make saved dir
    save_folder_name = os.path.join(args.save_dir, os.path.basename(args.ckpt_path))
    os.makedirs(save_folder_name, exist_ok=True)

    init_helper.init_logger(args.model_dir, args.log_file)
    logger.info(vars(args))

    # load model
    print('Loading DSNet model ...')
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)
    state_dict = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # load test videos
    sub_folders = os.listdir(args.test_source)

    # 依序存取每個子資料夾中的mp4檔案
    for i, sub_folder in enumerate(sub_folders):
        sub_folder_path = os.path.join(args.test_source, sub_folder)
        mp4_file_path = os.path.join(sub_folder_path, sub_folder + ".mp4")

        print('Preprocessing test video ...')
        video_proc = video_helper.VideoPreprocessor(args.sample_rate, args.backbone)
        n_frames, seq, cps, nfps, picks = video_proc.run(mp4_file_path)

        print('Predicting hit frames ...')
        with torch.no_grad():
            
            # 1 * n_frame * n_featureframes
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(args.device)

            pred_cls, pred_dr = model.predict(seq_torch)
            pred_cls /= pred_cls.max() + 1e-8

        # Peak detection
        pred_peaks = find_peaks(pred_cls, height=0.3, distance=15)[0]
        peak_count = len(pred_peaks)

        # 設置樣式
        sns.set_style("whitegrid")

        # 繪製散點圖
        ax = sns.scatterplot(x=pred_peaks, y=pred_cls[pred_peaks], color='red', label='y_peak')

        plt.plot(pred_cls, color='k')
        plt.legend()
        plt.text(5, 1.12, 'pred_count: ' + str(peak_count), fontsize=14)

        # plt.show()
        save_fig_path = os.path.join(save_folder_name, sub_folder + ".jpg")
        plt.savefig(save_fig_path, format='jpg')
        plt.close()

        save_csv_path = os.path.join(save_folder_name, sub_folder + ".csv")
        with open(save_csv_path, "w", newline="") as file:

            # Create a CSV writer object
            writer = csv.writer(file)

            # Write the list to the CSV file
            a = pred_peaks
            for row in a:
                writer.writerow([row])


if __name__ == '__main__':
    main()
