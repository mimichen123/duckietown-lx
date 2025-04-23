from typing import Tuple
import numpy as np
import cv2

def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """
    rows, cols = shape
    steer_matrix_left = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            # The further right the yellow line is detected, the more negative the steering (turn right)
            steer_matrix_left[i, j] = -1 * (j / cols)

    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """
    rows, cols = shape
    steer_matrix_right = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            # The further left the white line is detected, the more positive the steering (turn left)
            steer_matrix_right[i, j] = j / cols

    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Detects the lane markings in the input image.
    Returns:
        - mask_left_edge: Masked image for the dashed-yellow line (numpy.ndarray)
        - mask_right_edge: Masked image for the solid-white line (numpy.ndarray)
        - lane_center: The center point of the detected lane (Tuple[int, int])
    """
    # Convert the image to HSV color space for easier color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for yellow and white colors based on predefined RGB values and tolerances
    mask_left_edge = np.zeros_like(image[:, :, 0])  # Create a blank mask for yellow
    mask_right_edge = np.zeros_like(image[:, :, 0])  # Create a blank mask for white

    # Loop through all pixels in the image to detect the yellow and white lane markings
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            pixel = image[row, col]
            if is_color(pixel, yellow, YELLOW_TOLERANCE):  # Check for yellow
                mask_left_edge[row, col] = 255  # Mark this pixel as part of the yellow lane
            if is_color(pixel, white, WHITE_TOLERANCE):  # Check for white
                mask_right_edge[row, col] = 255  # Mark this pixel as part of the white lane

    # Find pixel coordinates of yellow and white features using the refined methods from the second code
    yellow, white, _ = analyze_img(image)

    # Calculate the lane center based on the detected yellow and white lane markings
    lane_center = None
    if yellow and white:
        lane_center = (yellow[0] + white[0]) / 2.0, (yellow[1] + white[1]) / 2.0
    elif white and not yellow:
        lane_center = (white[0], white[1] - white_width - 1*(LANE_WIDTH_PX / 2.0))
    elif yellow and not white:
        lane_center = (yellow[0], yellow[1] + yellow_width + 1*(LANE_WIDTH_PX / 2.0))

    # If no lane center was detected, use the previous center (you might need to store it for future use)
    if lane_center is None:
        lane_center = (0, 160)  # You can store and reuse the last lane center if needed

    return mask_left_edge, mask_right_edge, lane_center


def compute_steering(image: np.ndarray) -> float:
    """
    Computes the steering command based on the lane detection.
    
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    
    Return:
        steering: Steering command for the Duckiebot.
    """
    # Detect lane markings
    mask_left_edge, mask_right_edge = detect_lane_markings(image)
    
    # Get the weight matrices
    steer_matrix_left = get_steer_matrix_left_lane_markings(mask_left_edge.shape)
    steer_matrix_right = get_steer_matrix_right_lane_markings(mask_right_edge.shape)
    
    # Compute the steering using the weighted sum of detected lanes
    steering = np.sum(steer_matrix_left * mask_left_edge) + np.sum(steer_matrix_right * mask_right_edge)
    
    return steering
