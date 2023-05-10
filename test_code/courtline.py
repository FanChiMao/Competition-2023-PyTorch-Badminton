import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

"""
revise from: https://ramanlabs.in/static/tutorial/detecting_player_position_during_a_badminton_rally.html
"""

img = cv2.imread(r"D:\AICUP\Competition-2023-PyTorch-Badminton\test_code\court_sr_clean.png")
TOP_LEFT_2D = [124, 81]
TOP_RIGHT_2D = [493, 81]
MIDDLE_LEFT_2D = [89, 562]
MIDDLE_RIGHT_2D = [529, 562]
BOTTOM_LEFT_2D = [124, 1043]
BOTTOM_RIGHT_2D = [493, 1043]
NET_LEFT_2D = [89, 562]
NET_RIGHT_2D = [529, 562]
dst_coords = np.hstack((TOP_LEFT_2D, MIDDLE_LEFT_2D, BOTTOM_LEFT_2D, TOP_RIGHT_2D, MIDDLE_RIGHT_2D, BOTTOM_RIGHT_2D, NET_LEFT_2D, NET_RIGHT_2D)).reshape(8,2).astype("float32")

color = (255, 0, 0)
radius = 5
cv2.circle(img, MIDDLE_LEFT_2D, radius, color, -1)
cv2.circle(img, MIDDLE_RIGHT_2D, radius, color, -1)
cv2.circle(img, BOTTOM_LEFT_2D, radius, color, -1)
cv2.circle(img, BOTTOM_RIGHT_2D, radius, color, -1)
cv2.circle(img, TOP_LEFT_2D, radius, color, -1)
cv2.circle(img, TOP_RIGHT_2D, radius, color, -1)
cv2.imwrite("./test.png", img)

def get_coordinates(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coordinates: ({}, {})".format(x, y))


def find_court_corner():
    image = cv2.imread(r"D:\AICUP\Competition-2023-PyTorch-Badminton\test_code\court_sr_clean.png")
    # image = cv2.imread(r"D:\AICUP\datasets\gt_frames\video_0000_frame_000038.png")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_coordinates)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    find_court_corner()