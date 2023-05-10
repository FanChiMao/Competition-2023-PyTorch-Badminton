import numpy as np
from scipy.signal import find_peaks
import csv
import argparse
from tqdm import tqdm
from glob import glob
import os
import matplotlib.pyplot as plt
import math
import pandas as pd

def read_csv(path):
    x = []
    y = []
    frame = []
    front_zero = []
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        count = 0
        front_zero_count = 0
        previous_2 = '0'
        previous_1 = '0'
        for i, row in enumerate(rows):
            try:
                if row[1] == '1' and (previous_1 and previous_2) == '1' and count > 10:
                    x.append(int(row[2]))
                    y.append(int(row[3]))
                    frame.append(int(row[0]))
                    front_zero.append(front_zero_count)
                else:
                    front_zero_count += 1

                previous_2 = previous_1
                previous_1 = row[1]
                count += 1
            except:  # Header
                continue

    return x, y, frame, front_zero, count

def get_angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    angle1 = math.atan2(y2 - y1, x2 - x1)
    angle2 = math.atan2(y4 - y3, x4 - x3)
    angle = angle2 - angle1
    angle_degrees = math.degrees(angle)
    return angle_degrees


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder', type=str, default=r"./predict_train_csv",
                        help='The path includes predict csv files by TrackNet.')
    args = parser.parse_args()

    inp_dir = args.csv_folder

    csv_files = glob(os.path.join(inp_dir, '*.csv'))


    for i, csv_path in enumerate(csv_files):
        print(csv_path)
        x, y, z, front_zeros, total_frame = read_csv(csv_path)
        predict_hit_points = np.zeros(total_frame)
        ang = np.zeros(total_frame)

        plt.plot(z, y, '-')

        ## Basic shot
        peaks, _ = find_peaks(y, prominence=5, distance=10)

        ## Denoise begin and final
        if len(peaks) >= 5:
            lower = np.argmin(y[peaks[0]:peaks[1]])
            if (y[peaks[0]] - lower) < 5:
                peaks = np.delete(peaks, 0)
            lower = np.argmin(y[peaks[-2]:peaks[-1]])
            if (y[peaks[-1]] - lower) < 5:
                peaks = np.delete(peaks, -1)

        ## Adjust server time
        start_point = 0
        end_point = peaks[-1] + front_zeros[peaks[-1]]
        for j in range(len(y) - 1):
            if (y[j] - y[j + 1]) / (z[j + 1] - z[j]) >= 5:
                start_point = j + front_zeros[j]
                predict_hit_points[start_point] = 1
                # print(start_point)
                break

        for j in range(len(peaks)):
            print(peaks[j] + front_zeros[peaks[j]], end=', ')
            if (peaks[j] + front_zeros[peaks[j]]) >= start_point and peaks[j]+ front_zeros[peaks[j]] <= end_point:
                predict_hit_points[peaks[j]+front_zeros[peaks[j]]] = 1
        print("==> basic shot")

        # Remove first and final peaks
        # peaks = np.delete(peaks, 0)
        # peaks = np.delete(peaks, -1)
        # peaks = np.insert(peaks, 0, start_point)

        for j1 in range(len(peaks) - 1):
            start = peaks[j1]
            end = peaks[j1 + 1] + 1
            upper = []
            lower = np.argmin(y[start:end])

            for j2 in range(start + lower, end + 1):
                if j2 - (start + lower) > 5 and (end - j2) > 5:
                    if (y[j2] - y[j2-1])*3 < (y[j2+1] - y[j2]):
                        print(j2 + front_zeros[j2], end=', ')
                        ang[j2 + front_zeros[j2]] = 1

                    point = [x[j2], y[j2]]
                    line = [x[j2-1], y[j2-1], x[j2+1], y[j2+1]]
                    if get_angle([x[j2-1], y[j2-1], x[j2], y[j2]], [x[j2], y[j2], x[j2+1], y[j2+1]]) > 140:
                        print(j2 + front_zeros[j2], end=', ')
                        ang[j2 + front_zeros[j2]] = 1

        print("==> angle shot")
        ang, _ = find_peaks(ang, distance=15)

        for j in ang:
            predict_hit_points[j] = 1

        predict_hit_points, _ = find_peaks(predict_hit_points, distance=5)
        final_predict = []
        for j in predict_hit_points:
            print(j, end=', ')
            final_predict.append(j)
        print("==> final predict")
        print("=======================================================================================================")
        pass


