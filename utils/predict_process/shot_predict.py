import numpy as np
from scipy.signal import find_peaks
import csv
import argparse
from tqdm import tqdm
from glob import glob
import os
import matplotlib.pyplot as plt


def read_csv(path):
    x = []
    y = []
    frame = []
    front_zero = []
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        count = 0
        for row in rows:
            try:
                if row[1] == '1':
                    x.append(int(row[2]))
                    y.append(int(row[3]))
                    frame.append(int(row[0]))
                    front_zero.append(front_zero_index)
                else:
                    front_zero_index = count
                count += 1
            except:  # Header
                continue

    return x, y, frame, front_zero, count



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder', type=str, default=r"./predict_train",
                        help='The path includes predict csv files by TrackNet.')
    args = parser.parse_args()

    inp_dir = args.csv_folder

    csv_files = glob(os.path.join(inp_dir, '*.csv'))


    for i, csv_path in enumerate(tqdm(csv_files)):
        x, y, frame, front_zeros, total_frame = read_csv(csv_path)
        predict_hit_points = np.zeros(total_frame + 2)
        angle = np.zeros(total_frame + 2)

        plt.plot(frame, y, '-')

        peaks, _ = find_peaks(y, prominence=5)#, distance=10)

        if len(peaks) >= 5:
            lower = np.argmin(y[peaks[0]:peaks[1]])
            if (y[peaks[0]] - lower) < 5:
                peaks = np.delete(peaks, 0)
            lower = np.argmin(y[peaks[-2]:peaks[-1]])
            if (y[peaks[-1]] - lower) < 5:
                peaks = np.delete(peaks, -1)

        start_point = 0
        end_point = peaks[-1]
        for j in range(len(y) - 1):
            if (y[j] - y[j + 1]) / (frame[j + 1] - frame[j]) >= 5:
                start_point = j  # + front_zeros[j]
                predict_hit_points[start_point] = 1
                break

        for j in range(len(peaks)):
            print(peaks[j], end=', ')
            # if (peaks[j] + front_zeros[peaks[j]]) >= start_point and peaks[j]+ front_zeros[peaks[j]] <= end_point:
            #     predict_hit_points[peaks[j]+front_zeros[peaks[j]]] = 1
            if (peaks[j]) >= start_point and peaks[j] <= end_point:
                predict_hit_points[peaks[j]] = 1



        for j in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1] + 1
            upper = []
            lower = np.argmin(y[start:end])
            for j2 in range(start+lower, end+1):
                if j2-(start + lower) > 5 and (end - j2 > 5):
                    if y[j2] - y[j2 - 1]*3 < y[j2 + 1] - y[j2]:
                        angle[j2] = 1 #  + front_zeros[j2]
                    point = [x[j2], y[j2]]
                    line = [x[j2-1], y[j2-1], x[j2+1], y[j2+1]]

                    # if

        angle, _ = find_peaks(angle, distance=15)

        for j in angle:
            predict_hit_points[j] = 1

        predict_hit_points, _ = find_peaks(predict_hit_points, distance=5)
        final_predict = []
        for j in predict_hit_points:
            final_predict.append(j)


        pass


