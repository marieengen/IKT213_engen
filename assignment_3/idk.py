import cv2


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


if __name__ == "__main__":
    lambo = cv2.imread("lambo.png")
    sobel_edge_detection(lambo)
    canny_edge_detection(lambo, threshold_1=50, threshold_2=50)