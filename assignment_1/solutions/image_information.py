import cv2

def print_image_information(image):
    height, width, channels = image.shape
    size = image.size
    dtype = image.dtype
    print(f"height: {height}")
    print(f"width: {width}")
    print(f"channels: {channels}")
    print(f"size: {size}")
    print(f"data type: {dtype}")

def main():
    image = cv2.imread("iris-1.jpg")
    print_image_information(image)

if __name__ == "__main__":
    main()

import cv2

def save_camera_information(filepath="camera_outputs.txt"):
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    with open(filepath, "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

def main():
    save_camera_information()

if __name__ == "__main__":
    main()

