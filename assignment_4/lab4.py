import cv2
import numpy as np


def harris_corner_detection(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    output = reference_image.copy()
    output[dst > 0.01 * dst.max()] = [0, 0, 255]  # marker hjørner i rødt

    cv2.imwrite("harris.png", output)
    return output


def align_images(image_to_align, reference_image, max_features, good_match_percent):
    img_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(img_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(ref_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)

    # PS: den originale sort()-linjen fra LearnOpenCV-eksempelet er buggy,
    # bruker sorted() istedenfor, slik oppgaven ber om
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    match_img = cv2.drawMatches(
        image_to_align, keypoints1, reference_image, keypoints2, matches, None
    )
    cv2.imwrite("matches.png", match_img)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, _ = reference_image.shape
    aligned_img = cv2.warpPerspective(image_to_align, h, (width, height))

    cv2.imwrite("aligned.png", aligned_img)
    return aligned_img, match_img


if __name__ == "__main__":
    reference_img = cv2.imread("reference_img.png")
    align_this = cv2.imread("align_this.jpg")

    harris_corner_detection(reference_img)
    align_images(align_this, reference_img, max_features=1500, good_match_percent=0.15)