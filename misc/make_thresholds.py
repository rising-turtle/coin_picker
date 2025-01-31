import cv2
import numpy as np

thresholds = [(0, 100, 25, 127, -128, 127),    #Red
              (0, 100, -128, -18, 11, 127),    #Green
              (1, 100, -128, 127, -128, -20)]  #Blue

for t in thresholds:
    a = [t[0], t[2], t[4]]
    b = [t[1], t[3], t[5]]
    r1 = cv2.cvtColor(np.uint8([[a]]), cv2.COLOR_LAB2RGB)
    r2 = cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_LAB2RGB)
    print(list(r1[0][0]), list(r2[0][0]))