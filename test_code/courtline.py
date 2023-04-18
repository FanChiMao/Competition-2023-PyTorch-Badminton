import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

"""
revise from: https://ramanlabs.in/static/tutorial/detecting_player_position_during_a_badminton_rally.html
"""
def find_court_corner():
    image_path = r"D:\AICUP\Badminton\court_sr_clean.png"
    img = cv2.imread(image_path)



    # display the image
    cv2.imshow('Badminton Court', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def mapping_transformation():
    cap = cv2.VideoCapture(r"D:\AICUP\Badminton\dataset\Public\train")
    ret,frame = cap.read()





if __name__ == '__main__':
    # main()
    find_court_corner()
