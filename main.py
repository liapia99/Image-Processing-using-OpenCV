import numpy as np
import cv2

# Create a counter for debugging
circleCount = 0

# Read image
img = cv2.imread('hard-test-img.jpg', cv2.IMREAD_GRAYSCALE)

# Setting parameter values for Canny Edge filter
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

# Applying the Canny Edge filter
edge = cv2.Canny(img, t_lower, t_upper)

# Preparing/setting parameters for detecting circles
img = cv2.medianBlur(img, 5)
gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                           param1 = 50, param2 = 40, minRadius = 0, maxRadius = 0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    circleCount = circleCount + 1
    shape = 'Circle'
    # Drawing the outline
    cv2.circle(gray,(i[0],i[1]),i[2],(255,0,0),2)
    # Drawing the center of the circle
    cv2.circle(gray,(i[0],i[1]),2,(255,0,0),4)
    # Writing label "circle"
    cv2.putText(gray, shape, (i[0] - 30, i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# Opens images to show the progress of the image processing
cv2.imshow('Original', img)
cv2.imshow('Canny', edge)
cv2.imshow('Detected Circles', gray)
print("Circles:", circleCount)
cv2.waitKey(0)
cv2.destroyAllWindows()
