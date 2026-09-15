import cv2
import numpy as np


def sobel_edge_detection(image):
    blurred = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0)
    sobel = cv2.Sobel(blurred, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel_abs = cv2.convertScaleAbs(sobel)
    cv2.imwrite("sobel_edges.png", sobel_abs)
    return sobel_abs


def canny_edge_detection(image, threshold_1, threshold_2):
    blurred = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0)
    edges = cv2.Canny(blurred, threshold_1, threshold_2)
    cv2.imwrite("canny_edges.png", edges)
    return edges


def template_match(image, template):
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template_gray.shape

    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    locations = np.where(result >= threshold)

    output = image.copy()
    for pt in zip(*locations[::-1]):  # (x, y) par
        cv2.rectangle(output, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("template_match.png", output)
    return output


def resize(image, scale_factor: int, up_or_down: str):
    result = image.copy()
    steps = int(np.log2(scale_factor))

    for _ in range(steps):
        if up_or_down == "up":
            result = cv2.pyrUp(result)
        elif up_or_down == "down":
            result = cv2.pyrDown(result)
        else:
            raise ValueError("up_or_down må være 'up' eller 'down'")

    cv2.imwrite("resized.png", result)
    return result


if __name__ == "__main__":
    lambo = cv2.imread("lambo.png")
    sobel_edge_detection(lambo)
    canny_edge_detection(lambo, threshold_1=50, threshold_2=50)
    resize(lambo, scale_factor=2, up_or_down="up")

    shapes = cv2.imread("shapes.png")
    shapes_template = cv2.imread("shapes_template.jpg")
    template_match(shapes, shapes_template)