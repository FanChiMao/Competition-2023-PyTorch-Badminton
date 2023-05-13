import cv2


image = cv2.imread(r'D:\AICUP\datasets\gt_frames\video_0009_frame_000041.png')

radius = 5


cv2.circle(image, (353,392), radius, (255, 0, 0), -1) # GT


cv2.imwrite('./frame_042.png', image)

