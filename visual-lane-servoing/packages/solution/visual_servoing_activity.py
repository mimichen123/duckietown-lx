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


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
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

    return mask_left_edge, mask_right_edge


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
