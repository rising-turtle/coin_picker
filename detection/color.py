"""(NOT FOR PRODUCTION) Coin classification by color with manual thresholding using OpenCV"""

__author__ = "Yina Tang"


import cv2
import numpy as np


def color_stuff(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # HSV values for copper in ideal lighting conditions: 24 49% 86%; 24 47% 91%; 24 52% 69%; 24 55% 53%
    # Calculated average hue: 13
    lower_copper = np.array([10, 20, 48])  # penny
    upper_copper = np.array([14, 180, 150])

    # HSV values for silver in ideal lighting conditions: 42 13% 91%; 42 19% 53%; 46 20% 32%; 46 22% 29%
    # Calculated average hue: [20.8, 23.3]
    lower_silver = np.array([0, 0, 0])  # nickel
    upper_silver = np.array([179, 128, 128])  # average hue
    # upper_silver = np.array([179, 128, 255])  # with this one you only get the edges

    # ignore dime and quarter until we have depth data (copper edges)

    mask_copper = cv2.cvtColor(cv2.inRange(hsv, lower_copper, upper_copper), cv2.COLOR_BGR2GRAY)
    mask_silver = cv2.cvtColor(cv2.inRange(hsv, lower_silver, upper_silver), cv2.COLOR_BGR2GRAY)
    mask = mask_silver  # for testing purposes

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    def get_average_hue(image, mask):
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        total = cv2.countNonZero(mask)
        if total == 0:
            return 0

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_image[:, :, 0]
        average_hue = cv2.sumElems(hue_channel)[0] / total

        return average_hue


    average_hue = get_average_hue(image, mask)
    print(f"Average hue: {average_hue}")



# Show edges from Canny
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])


# Show masked images
# plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(cv2.cvtColor(mask_penny, cv2.COLOR_BGR2RGB))
# plt.title('Mask'), plt.xticks([]), plt.yticks([])

# plt.show()

# cv2.imshow('Original Image', image)
# cv2.imshow('Mask', mask)
# cv2.imshow('Masked Image', masked_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
