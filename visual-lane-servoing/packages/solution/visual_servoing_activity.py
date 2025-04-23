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

    # Define color ranges for yellow dashed line and white solid line
    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([180, 25, 255], dtype=np.uint8)

    # Create masks for yellow and white colors
    mask_left_edge = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    mask_right_edge = cv2.inRange(hsv_image, white_lower, white_upper)

    # Find the coordinates of the left (yellow) and right (white) lane markings
    yellow = None
    white = None

    # Find yellow line (left lane)
    yellow_contours, _ = cv2.findContours(mask_left_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if yellow_contours:
        yellow = max(yellow_contours, key=cv2.contourArea)  # Get the largest contour (yellow line)
        yellow_moments = cv2.moments(yellow)
        yellow_center_x = int(yellow_moments['m10'] / yellow_moments['m00'])
        yellow_center_y = int(yellow_moments['m01'] / yellow_moments['m00'])
        yellow = (yellow_center_x, yellow_center_y)

    # Find white line (right lane)
    white_contours, _ = cv2.findContours(mask_right_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if white_contours:
        white = max(white_contours, key=cv2.contourArea)  # Get the largest contour (white line)
        white_moments = cv2.moments(white)
        white_center_x = int(white_moments['m10'] / white_moments['m00'])
        white_center_y = int(white_moments['m01'] / white_moments['m00'])
        white = (white_center_x, white_center_y)

    # Calculate the lane center based on the detected yellow and white lane markings
    lane_center = None
    if yellow and white:
        lane_center = ((yellow[0] + white[0]) / 2.0, (yellow[1] + white[1]) / 2.0)
    elif white and not yellow:
        lane_center = (white[0], white[1] - 100)  # Adjusting based on white line (for example)
    elif yellow and not white:
        lane_center = (yellow[0], yellow[1] + 100)  # Adjusting based on yellow line (for example)

    # If no lane center was detected, use a default center (or the previous one stored)
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
