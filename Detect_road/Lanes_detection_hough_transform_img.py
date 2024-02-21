import cv2
import numpy as np
import matplotlib.pyplot as plt

df = "D:\CODE\DEEP_LEARNING__COMPUTER_VISION\Computer_vision\OpenCV4\Project\Detect_road\data"

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    print(img.shape)
    y1 = img.shape[0]
    y2 = int(y1 * (3/4))
    x1 = int((y1 - intercept)/ slope)
    x2 = int((y2 - intercept)/ slope)
    return np.array([x1, y1, x2, y2])

 
def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
            
        else:
            right_fit.append((slope, intercept))
    # if len(left_fit) and len(right_fit):
    # ##over-simplified if statement (should give you an idea of why the error occurs)
    #     left_fit_average  = np.average(left_fit, axis=0)
    #     right_fit_average = np.average(right_fit, axis=0)
    #     left_line  = make_coordinates(img, left_fit_average)
    #     right_line = make_coordinates(img, right_fit_average)
    #     averaged_lines = [left_line, right_line]
    #     return averaged_lines
    left_fit_average = np.average(left_fit, axis= 0)
    right_fit_average = np.average(right_fit, axis= 0)  
    left_line = make_coordinates(img, left_fit_average)
    right_line = make_coordinates(img, right_fit_average)
    return np.array([left_line, right_line])


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blur, 50, 120)
    return canny

# bên cạnh dùng cách này ta có thể dùng thế này
# def display_lines(img, lines):
#     line_img = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             print(line)
#             x1, y1, x2, y2 = line.reshape(4)
#             cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
#     return line_img


def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return line_img

# hàm này dùng để tạo ra hình tam giác và giúp canny có thể detect ra dc vùng mà mình cần để detection
def region_of_interest(img): 
    height = img.shape[0]
    polygons = np.array([
        [(290, height), (890, height), (620, 500)]
        ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img



img = cv2.imread(df + "\lane3.jpg")
lane_img = np.copy(img)
Canny_img = canny(lane_img)
cropped_img = region_of_interest(Canny_img)

lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 50, np.array([]),minLineLength=15, maxLineGap=2)
average_img = average_slope_intercept(lane_img, lines)
line_img = display_lines(lane_img, average_img)
combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
plt.imshow(line_img)
plt.show()


# plt.imshow(combo_img)
# plt.show()

# cv2.imshow("result", combo_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

