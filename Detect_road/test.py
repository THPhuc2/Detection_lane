def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    print(img.shape)
    y1 = img.shape[0]
    y2 = int(y1 * (3/5)) # 540
    x1 = int((y1 - intercept)/ slope)
    x2 = int((y2 - intercept)/ slope)
    # x2 = int(x1 - (y1 - y2) / slope)

    print(f"y2: {y2}, x1: {x1}, x2: {x2}, slope: {slope}, intercept: {intercept}")
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
    if len(left_fit) and len(right_fit):
    ##over-simplified if statement (should give you an idea of why the error occurs)
        left_fit_average  = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line  = make_coordinates(img, left_fit_average)
        right_line = make_coordinates(img, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines
    # print(left_fit, "left")
    # print(right_fit, "right")
    # left_fit_average = np.average(left_fit, axis= 0)
    # right_fit_average = np.average(right_fit, axis= 0)
    # print(left_fit_average, "left")
    # print(right_fit_average, "right")    
    # left_line = make_coordinates(img, left_fit_average)
    # right_line = make_coordinates(img, right_fit_average)
    # print(left_line, "left")
    # print(right_line, "right") 
    # return np.array([left_line, right_line])