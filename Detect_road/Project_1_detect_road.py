
import cv2
import numpy as np
import matplotlib.pylab as plt
import os

df = "D:\\CODE\\DEEP_LEARNING__COMPUTER_VISION\\Computer_vision\\OpenCV4\\Picture"
img = cv2.imread(df + "\\road1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cap = cv2.VideoCapture(df + "\\Manhattan_Trim.mp4")
while cap.isOpened():
    _, frame = cap.read()
    print(frame.shape)
# # detect road in image

# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     # channel_count = img.shape[2]
#     # match_mask_color = (255,) * channel_count
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color) # vẽ da giác trắng (các đỉnh dc xác định bởi vertices)
#     masked_image = cv2.bitwise_and(img, mask)  # giữ lại các pixel trong vùng trung tâm
#     return masked_image

# def draw_the_lines(img, lines):
#     img = np.copy(img)
#     blank_img = np.zeros((img.shape[0], img.shape[1], 3), dtype= np.uint8)

#     for line in lines:
#         for x1, y1, x2, y2  in line:
#             cv2.line(blank_img, (x1, y1), (x2, y2), (255, 0, 0), 4)

#     img = cv2.addWeighted(img, 0.8, blank_img, 1, 0.0)

#     return img
# print(img.shape)

# height = img.shape[0]
# width = img.shape[1]

# # danh sách các đinh quan tâm
# region_of_interest_vertices = [
#     (0, height),
#     (width/2, height/2),
#     (width, height)
# ]

# gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# gray_img = cv2.dilate(gray_img, kernel= np.ones((5, 5), np.uint8))

# canny_img = cv2.Canny(gray_img, 100, 200)
# cropped_img = region_of_interest(canny_img,
#                                  np.array([region_of_interest_vertices], np.int32),)


# lines = cv2.HoughLinesP(cropped_img, 6, np.pi/60,
#                         160, np.array([]),
#                         40, 25)
# img_with_lines = draw_the_lines(img, lines)



# plt.subplot(121), plt.imshow(cropped_img)
# plt.subplot(122), plt.imshow(img_with_lines)
# plt.show()

