import cv2 as cv
import numpy as np

# Read the main image
src = inputImage = cv.imread('curve.jpg')
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

def find_bezier_curves(contours, min_vertices):
    bezier_curves = []
    for contour in contours:
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        if len(approx) > min_vertices:
            bezier_curves.append(contour)
    return bezier_curves

def draw_lines(image, lines, color=(0, 0, 255), thickness=4):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), color, thickness)

def draw_bezier_curves(image, curves, color=(255, 0, 255), thickness=6):
    for curve in curves:
        cv.drawContours(image, [curve], 0, color, thickness)

def draw_circles(image, circles, color=(255, 0, 0), thickness=4):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv.circle(image, (circle[0], circle[1]), circle[2], color, thickness)

def thresh_callback(val):
    threshold = val
    # Detect edges using Canny with a lower threshold
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find Bezier curves
    bezier_curves = find_bezier_curves(contours, min_vertices=5)

    # Draw contours for all shapes
    contours_image = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    color = (0, 255, 0)  # Green for all contours
    for i in range(len(contours)):
        cv.drawContours(contours_image, contours, i, color, 2, cv.LINE_AA)

    # Draw contours for Bezier curves
    bezier_color = (255, 0, 255)  # Yellow for Bezier curves
    draw_bezier_curves(contours_image, bezier_curves, bezier_color)

    # Show contours image
    cv.imshow('Contours', contours_image)

    # Draw straight lines
    lines = cv.HoughLinesP(canny_output, 1, np.pi / 180, threshold=65, minLineLength=0, maxLineGap=0)
    lines_image = np.copy(src)
    draw_lines(lines_image, lines)

    # Show lines image
    cv.imshow('Lines', lines_image)

    # Draw circles
    circles = cv.HoughCircles(src_gray, cv.HOUGH_GRADIENT, 1, 10,
                              param1=40, param2=80, minRadius=0, maxRadius=0)
    circles_image = np.copy(src)
    draw_circles(circles_image, circles)

    # Show circles image
    cv.imshow('Circles', circles_image)

    # Draw all shapes together
    combined_image = np.copy(src)
    draw_lines(combined_image, lines, color=(0, 0, 255))  # Red for lines
    draw_bezier_curves(combined_image, bezier_curves)
    draw_circles(combined_image, circles)

    # Show combined image
    cv.imshow('Combined Shapes', combined_image)


# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

max_thresh = 255
thresh = 100  # Lower initial threshold

thresh_callback(thresh)

cv.waitKey(0)
cv.destroyAllWindows()
