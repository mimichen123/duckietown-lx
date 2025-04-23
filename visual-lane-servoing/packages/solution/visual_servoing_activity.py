from typing import Tuple
import numpy as np
import cv2

# Define the color parameters
yellow = (254, 240, 82, 35)
white = (254, 255, 253, 40)

# Tolerances for color detection
YELLOW_TOLERANCE = 75
WHITE_TOLERANCE = 40

# Image size
img_width = 320
img_height = 240

# Lane width constants (as an example)
LANE_WIDTH_PX = 240

# Function to check if a pixel matches the specified color
def is_color(pixel, color, tolerance):
    return abs(pixel[0] - color[0]) < tolerance and \
           abs(pixel[1] - color[1]) < tolerance and \
           abs(pixel[2] - color[2]) < tolerance

# Function to analyze image and detect lane features
def analyze_img(image):
    yellow_px = None
    white_px = None

    # Iterate through the image to find yellow and white pixels
    for i in range(0, image.shape[0], 4):  # Loop through rows
        for j in range(0, image.shape[1], 4):  # Loop through columns
            pixel = image[i, j]
            if is_color(pixel, yellow, YELLOW_TOLERANCE) and yellow_px is None:
                yellow_px = (i, j)
            if is_color(pixel, white, WHITE_TOLERANCE) and white_px is None:
                white_px = (i, j)

            # Break the loop once both yellow and white are found
            if yellow_px and white_px:
                break
        if yellow_px and white_px:
            break

    return yellow_px, white_px, None  # No red detected in this example

# Function to get the steering matrix for the left lane (yellow line)
def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    steer_matrix_left = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            steer_matrix_left[i, j] = -1 * (j / cols)

    return steer_matrix_left

# Function to get the steering matrix for the right lane (white line)
def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    steer_matrix_right = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            steer_matrix_right[i, j] = j / cols

    return steer_matrix_right

# Main function to detect lane markings and compute lane center
def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create blank masks for yellow and white lanes
    mask_left_edge = np.zeros_like(image[:, :, 0])
    mask_right_edge = np.zeros_like(image[:, :, 0])

    # Detect yellow and white lane markings
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            pixel = image[row, col]
            if is_color(pixel, yellow, YELLOW_TOLERANCE):
                mask_left_edge[row, col] = 255
            if is_color(pixel, white, WHITE_TOLERANCE):
                mask_right_edge[row, col] = 255

    # Find pixel coordinates of yellow and white lane markings
    yellow, white, _ = analyze_img(image)

    # Calculate the lane center
    lane_center = None
    if yellow and white:
        lane_center = (yellow[0] + white[0]) / 2.0, (yellow[1] + white[1]) / 2.0
    elif white and not yellow:
        lane_center = (white[0], white[1] - (LANE_WIDTH_PX / 2.0))
    elif yellow and not white:
        lane_center = (yellow[0], yellow[1] + (LANE_WIDTH_PX / 2.0))

    if lane_center is None:
        lane_center = (0, 160)  # Fallback if no lane center is detected

    return mask_left_edge, mask_right_edge, lane_center

# Compute the steering command based on the lane detection
def compute_steering(image: np.ndarray) -> float:
    mask_left_edge, mask_right_edge, lane_center = detect_lane_markings(image)

    # Get the steering matrices
    steer_matrix_left = get_steer_matrix_left_lane_markings(mask_left_edge.shape)
    steer_matrix_right = get_steer_matrix_right_lane_markings(mask_right_edge.shape)

    # Calculate steering using the weighted sum of left and right lane masks
    steering = np.sum(steer_matrix_left * mask_left_edge) + np.sum(steer_matrix_right * mask_right_edge)

    return steering
