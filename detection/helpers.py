import cv2
from cv2.typing import MatLike
from PIL import Image, ImageFile

def resized(image: MatLike) -> MatLike:
    """Resizes an OpenCV or a PIL image to output size of RealSense D405 (1280 × 720) while maintaining aspect ratio.
    Automatically checks whether the image is a PIL Image or an OpenCV image and resizes it accordingly.

    Side benefit: removes noise from image for better edge detection.

    Reference: https://www.intelrealsense.com/depth-camera-d405/
    """

    if isinstance(image, ImageFile.ImageFile):
        raise TypeError("Use resizedImageFile for PIL Images.")

    if not isinstance(image, MatLike):
        raise TypeError("Input must be an OpenCV image.")

    MAX_SIZE = (1280, 720)

    height, width = image.shape[:2]
    aspect_ratio = width / height

    if width > height:
        new_width = MAX_SIZE[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = MAX_SIZE[1]
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image


def resizedImageFile(image: ImageFile.ImageFile, outpath=None) -> ImageFile.ImageFile:
    """Resizes a PIL image to output size of RealSense D405 (1280 × 720) while maintaining aspect ratio.
    Checks whether the image is a PIL Image resizes it accordingly.

    Side benefit: removes noise from image for better edge detection.

    Reference: https://www.intelrealsense.com/depth-camera-d405/
    """

    if not isinstance(image, ImageFile.ImageFile):
        raise TypeError("Input must be a PIL Image.")

    MAX_SIZE = (1280, 720)
    image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
    if outpath != None: image.save(outpath)
    return image
