import cv2
import numpy as np

# The coordinates of the four corners of the badminton court in the image
# Replace these with the actual coordinates from your image
if __name__ == '__main__':
    import numpy as np

    # Your data
    x = [[746,201], [925, 390], [593, 336], [925, 390], [527, 355]]
    y = [[573, 466], [694, 318], [594, 494], [605, 379], [525, 441]]

    # Flatten the lists
    x = [item for sublist in x for item in sublist]
    y = [item for sublist in y for item in sublist]

    # Calculate the linear regression
    s1, s2, s3 = np.polyfit(x, y, 2)


    x = [353,392]
    y1 = s1 * x[0]**2 + s2 * x[0]**1 + s3
    y2 = s1 * x[1]**2 + s2 * x[1]**1 + s3

    print(f"{y1}, {y2}")