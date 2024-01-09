import cv2 as cv
import numpy as np
import argparse

# Read the main image
src = inputImage = cv.imread('curve.jpg')
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

def thresh_callback(val):
    threshold = val
    # Detect edges using Canny with a lower threshold
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    color = (0, 255, 0)  # Green for all contours
    for i in range(len(contours)):
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_AA)

    # Show in a window
    cv.imshow('Contours', drawing)

    # Hough Lines (Probabilistic) detection and drawing
    lines = cv.HoughLinesP(canny_output, 1, np.pi / 180, threshold=65, minLineLength=0, maxLineGap=10)
    if lines is not None:
        for idx, line in enumerate(lines):
            shape = f'Line {idx + 1}'
            x1, y1, x2, y2 = line[0]
            cv.line(drawing, (x1, y1), (x2, y2), (0, 0, 255), 4)
            # Writing label "Line"
         #   cv.putText(drawing, shape, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Hough Circles detection and drawing
    circles = cv.HoughCircles(src_gray, cv.HOUGH_GRADIENT, 1, 10,
                              param1=40, param2=80, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i, circle in enumerate(circles[0, :]):
            shape = f'Circle {i + 1}'
            color = (255, 0, 0)  # Blue for circles
            # Drawing the outline
            cv.circle(drawing, (circle[0], circle[1]), circle[2], color, 6)
            # Writing label "circle"
            cv.putText(drawing, shape, (circle[0] - 30, circle[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # Show in a window
    cv.imshow('Contours with Lines and Circles', drawing)


# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

max_thresh = 255
thresh = 100  # Lower initial threshold

thresh_callback(thresh)

cv.waitKey(0)
cv.destroyAllWindows()