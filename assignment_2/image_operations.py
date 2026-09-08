import cv2
import numpy as np

def padding(image, border_width):
    padded = cv2.copyMakeBorder(image, border_width, border_width,
                                border_width, border_width,
                                cv2.BORDER_REFLECT)
    cv2.imwrite("padded.png", padded)
    return padded

def crop(image, x_0, x_1, y_0, y_1):
    cropped = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("cropped.png", cropped)
    return cropped

def resize(image, width, height):
    resized = cv2.resize(image, (width, height))
    cv2.imwrite("resized.png", resized)
    return resized

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]
    cv2.imwrite("copied.png", emptyPictureArray)
    return emptyPictureArray

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale.png", gray)
    return gray

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv.png", hsv_image)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                value = int(image[y, x, c]) + hue
                emptyPictureArray[y, x, c] = np.clip(value, 0, 255)
    cv2.imwrite("hue_shifted.png", emptyPictureArray)
    return emptyPictureArray

def smoothing(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite("smoothed.png", blurred)
    return blurred

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("rotation_angle must be 90 or 180")
    cv2.imwrite(f"rotated_{rotation_angle}.png", rotated)
    return rotated

def main():
    image = cv2.imread("iris-1.png")
    height, width, channels = image.shape

    padding(image, 100)
    crop(image, 200, width - 130, 200, height - 130)
    resize(image, 200, 200)
    copy(image, np.zeros((height, width, 3), dtype=np.uint8))
    grayscale(image)
    hsv(image)
    hue_shifted(image, np.zeros((height, width, 3), dtype=np.uint8), 50)
    smoothing(image)
    rotation(image, 90)
    rotation(image, 180)

if __name__ == "__main__":
    main()