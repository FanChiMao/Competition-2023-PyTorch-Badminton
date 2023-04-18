import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

"""
revise from: https://ramanlabs.in/static/tutorial/detecting_player_position_during_a_badminton_rally.html
"""
LT_2D = [86, 78]
RT_2D = [532, 78]
LM_2D = [86, 562]
RM_2D = [532, 562]
LB_2D = [86, 1046]
RB_2D = [532, 1046]
dst_coords = np.hstack((LT_2D, RT_2D, LM_2D, RM_2D, LB_2D, RB_2D)).reshape(6,2).astype("float32")

LT = [427, 384]
RT = [861, 384]
LM = [379, 486]
RM = [908, 486]
LB = [286, 670]
RB = [1001, 670]

src_coords = np.hstack((LT, RT, LM, RM, LB, RB)).reshape(6, 2)
src_coords_final = np.hstack((src_coords, np.ones((src_coords.shape[0], 1)))).astype("float32")

def display_point():
    frame = cv2.imread(r"D:\AICUP\Badminton\dataset\Public\train_frames\v_0000_frame_000001.png")
    for (x, y) in src_coords:
        cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    plt.figure(figsize=(10, 10))
    plt.imshow(frame[:, :, ::-1])
    plt.show()



def get_coordinates(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coordinates: ({}, {})".format(x, y))


def find_court_corner():
    # image = cv2.imread(r"D:\AICUP\Badminton\court_sr_clean.png")
    image = cv2.imread(r"D:\AICUP\Badminton\dataset\Public\train_frames\v_0000_frame_000001.png")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_coordinates)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mapping_transformation():
    M_transform = lstsq(src_coords_final[:3, :], dst_coords[:3, :], rcond=-1)[0]
    print(M_transform)


if __name__ == '__main__':
    # find_court_corner()
    # display_point()
    mapping_transformation()
