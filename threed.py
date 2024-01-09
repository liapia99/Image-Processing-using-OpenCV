import cv2 as cv
import numpy as np
from mayavi import mlab

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


def plot_3d_model(circles, lines, bezier_curves):
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600))

    # Plot circles as spheres
    for circle in circles[0]:
        x, y, r = circle
        mlab.points3d(x, y, 0, scale_factor=r, color=(0, 0, 1), resolution=50)

    # Plot lines as cylinders
    for line in lines:
        x1, y1, x2, y2 = line[0]
        mlab.plot3d([x1, x2], [y1, y2], [0, 0], tube_radius=0.1, color=(1, 0, 0))

    # Plot bezier curves as tube-like representation
    for curve in bezier_curves:
        x = curve[:, 0, 0]
        y = curve[:, 0, 1]
        z = np.zeros_like(x)

        for i in range(len(x) - 1):
            mlab.plot3d([x[i], x[i + 1]], [y[i], y[i + 1]], [0, 0], tube_radius=0.1, color=(0, 1, 0))

    mlab.show()


def thresh_callback(val):
    threshold = val
    # Detect edges using Canny with a lower threshold
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find circles
    circles = cv.HoughCircles(src_gray, cv.HOUGH_GRADIENT, 1, 10,
                              param1=40, param2=80, minRadius=0, maxRadius=0)

    # Find lines
    lines = cv.HoughLinesP(canny_output, 1, np.pi / 180, threshold=65, minLineLength=0, maxLineGap=10)

    # Find Bezier curves
    bezier_curves = find_bezier_curves(contours, min_vertices=5)

    # Plot 3D model
    plot_3d_model(circles, lines, bezier_curves)


# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

max_thresh = 255
thresh = 100  # Lower initial threshold

thresh_callback(thresh)

cv.waitKey(0)
cv.destroyAllWindows()
