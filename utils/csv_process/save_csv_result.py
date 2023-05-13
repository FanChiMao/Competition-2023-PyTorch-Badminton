from tqdm import tqdm
import os
import csv
"""
 0: VideoName
 1: ShotSeq
 2: HitFrame 
 3: Hitter　(A or B)
 4: BallHeight　(1 or 2)
 5: RoundHead (1 or 2)
 6: Backhand (1 or 2)
 7: LandingX 
 8: LandingY 
 9: HitterLocationX 
10: HitterLocationY
11: DefenderLocationX
12: DefenderLocationY 
13: BallType (1 ~ 9)
14 :Winner
"""

csv_header = ["VideoName","ShotSeq","HitFrame","Hitter","BallHeight","RoundHead","Backhand","LandingX","LandingY",
              "HitterLocationX","HitterLocationY","DefenderLocationX","DefenderLocationY","BallType","Winner"]


def write_result_csv(save_path, result):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, 'result.csv')
    print(f"==> Start to write csv to {path}")
    with open(path, 'w', newline='') as csvfile1:
        h = csv.writer(csvfile1)
        h.writerow(csv_header)
        for i, line in enumerate(tqdm(result)):
            h.writerow(line)


if __name__ == "__main__":
    write_result_csv('./test', '')
