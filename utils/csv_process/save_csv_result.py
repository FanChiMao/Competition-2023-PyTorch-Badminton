import os



csv_header = r"VideoName,ShotSeq,HitFrame,Hitter,BallHeight,RoundHead,Backhand,LandingX,LandingY," \
             r"HitterLocationX,HitterLocationY,DefenderLocationX,DefenderLocationY,BallType,Winner\n"


def write_result_csv(video_path, save_path, predict):
    video_name = os.path.basename(video_path)
    save_path = os.path.join(save_path, video_name[:-4]+'.csv')

    with open(save_path, 'w'):
        f.write(csv_header)




if __name__ == "__main__":
    pass
