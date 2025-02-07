"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    cropped_img = cv2.bitwise_and(image, mask)
    return cropped_img


def draw_lines(image, hough_lines):
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


# img = cv2.imread("saved_frame.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process(img):
    height = img.shape[0]
    width = img.shape[1]
    roi_vertices = [
        (0, 650),
        (2*width/3, 2*height/3),
        (width, 1000)
    ]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.dilate(gray_img, kernel=np.ones((3, 3), np.uint8))

    canny = cv2.Canny(gray_img, 130, 220)

    roi_img = roi(canny, np.array([roi_vertices], np.int32))

    lines = cv2.HoughLinesP(roi_img, 2, np.pi / 180, threshold=10, minLineLength=15, maxLineGap=2)

    final_img = draw_lines(img, lines)

    return final_img


df = "D:\\CODE\\DEEP_LEARNING__COMPUTER_VISION\\Computer_vision\OpenCV4\\Project\\Detect_road\\data"
# cap = cv2.VideoCapture(df + "\\lane_vid2.mp4")
cap = cv2.VideoCapture(df + "\\Manhattan_Trim.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
saved_frame = cv2.VideoWriter("lane_detection.avi", fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    print(frame.shape)

    try:
        frame = process(frame)

        saved_frame.write(frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        break

cap.release()
saved_frame.release()
cv2.destroyAllWindows()
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define ROI
def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


# Draw Hough Lines on image
def draw_lines(lines, image):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


def process(img):
    try:

        # Define roi vertices
        h, w, _ = img.shape
        roi_vertices = [
            (200, h),
            (w / 2, 2 * h / 3),
            (w - 100, h)
        ]

        # Convert to GRAYSCALE
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply dilation (morphology)
        kernel = np.ones((3, 3), np.uint8)
        gray_img = cv2.dilate(gray_img, kernel=kernel)

        # Canny edge detection
        canny = cv2.Canny(gray_img, 60, 255)

        # ROI
        roi_image = roi(canny, np.array([roi_vertices], np.int32))

        # Hough Lines
        hough_lines = cv2.HoughLinesP(roi_image, 1, np.pi / 180, 40, minLineLength=10, maxLineGap=5)
        final_img = draw_lines(hough_lines, img)
        return final_img

    except Exception:
        return img


df = "D:\\CODE\\DEEP_LEARNING__COMPUTER_VISION\\Computer_vision\OpenCV4\\Project\\Detect_road\\data"
cap = cv2.VideoCapture(df + "\\lane_vid2.mp4")
# cap = cv2.VideoCapture(df + "\\Manhattan_Trim.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
saved_frame = cv2.VideoWriter("Manhattan_detection.avi", fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    _, frame = cap.read()

    try:

        frame = process(frame)
        saved_frame.write(frame)
        cv2.imshow("final", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        break

cap.release()
saved_frame.release()
cv2.destroyAllWindows()
