import cv2
import numpy as np
import matplotlib.pylab as plt
import os
# detect road in video


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    # match_mask_color = (255,) * channel_count
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color) # vẽ da giác trắng (các đỉnh dc xác định bởi vertices)
    masked_image = cv2.bitwise_and(img, mask)  # giữ lại các pixel trong vùng trung tâm
    return masked_image

# def draw_the_lines(img, lines):
#     img = np.copy(img)
#     blank_img = np.zeros((img.shape[0], img.shape[1], 3), dtype= np.uint8)
#
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             cv2.line(blank_img, (x1, y1), (x2, y2), (255, 255, 0), 3)
#
#     img = cv2.addWeighted(img, 0.8, blank_img, 1, 0.0)
#
#     return img

def draw_the_lines(img, vertices):
    for line in vertices:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return img

def process(img):
    height = img.shape[0]
    width = img.shape[1]

    # danh sách các đinh quan tâm
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.dilate(gray_img, kernel= np.ones((3, 3), np.uint8))

    canny_img = cv2.Canny(gray_img, 60, 250)
    cropped_img = region_of_interest(canny_img,
                                     np.array([region_of_interest_vertices], np.int32))


    lines = cv2.HoughLinesP(cropped_img, 1, np.pi/180,
                            160, np.array([]),
                            40, 20)
    img_with_lines = draw_the_lines(img, lines)
    return img_with_lines

df = "D:\\CODE\\DEEP_LEARNING__COMPUTER_VISION\\Computer_vision\OpenCV4\\Project\\Detect_road\\data"
# cap = cv2.VideoCapture(df + "\\lane_vid2.mp4")
cap = cv2.VideoCapture(df + "\\Manhattan_Trim.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
saved_frame = cv2.VideoWriter("Manhattan_detection.avi", fourcc, 30.0, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    saved_frame.write(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
saved_frame.release()
cv2.destroyAllWindows()

