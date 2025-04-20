import time
import numpy as np
import cv2
from enum import Enum
from PIL import Image
from typing import Tuple
import cv  # Duckietown's computer vision library

class DriveState(Enum):
    STOPPED = 1
    FOLLOWING_LANE = 2
    FINISHED = 3

class Driver:
    def __init__(self, car_interface):
        self.car = car_interface  # Duckiebot car interface
        self.speed_limit = 10.0
        self.state = DriveState.FOLLOWING_LANE
        self.num_updates = 0
        self.start_time = time.time()

    def update(self, image):
        if self.state == DriveState.FOLLOWING_LANE:
            steering = compute_steering(image)
            # Use the computed steering to control the car's motors
            self.car.command_motor_pwms(self.speed_limit, steering)

    def debug_image(self, image):
        print('Debugging image')
        # Draw region of interest (ROI) for debugging purposes
        cv.draw_region(image, cv.red_roi, (255, 255, 255))
        cv.draw_region(image, cv.green_roi, (255, 255, 0))
        Image.fromarray(image, 'RGB').show()

# Function to compute the steering based on lane markings
def compute_steering(image: np.ndarray) -> float:
    mask_left_edge, mask_right_edge = detect_lane_markings(image)
    
    # Get the steering weight matrices for left and right lane markings
    steer_matrix_left = get_steer_matrix_left_lane_markings(mask_left_edge.shape)
    steer_matrix_right = get_steer_matrix_right_lane_markings(mask_right_edge.shape)
    
    # Compute the steering command by summing weighted masks
    steering = np.sum(steer_matrix_left * mask_left_edge) + np.sum(steer_matrix_right * mask_right_edge)
    
    return steering

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects the lane markings in the input image.
    Returns masks for yellow (left) and white (right) lane markings.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([180, 25, 255], dtype=np.uint8)

    mask_left_edge = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    mask_right_edge = cv2.inRange(hsv_image, white_lower, white_upper)

    return mask_left_edge, mask_right_edge

def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape: The shape of the steer matrix.
    Returns:
        steer_matrix_left: The steering matrix for Braitenberg-like control using the left lane markings.
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
        shape: The shape of the steer matrix.
    Returns:
        steer_matrix_right: The steering matrix for Braitenberg-like control using the right lane markings.
    """
    rows, cols = shape
    steer_matrix_right = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            # The further left the white line is detected, the more positive the steering (turn left)
            steer_matrix_right[i, j] = j / cols

    return steer_matrix_right

if __name__ == '__main__':
    # Initialize car interface (e.g., use a mock or real car control interface)
    mock_car = MockCar()
    driver = Driver(mock_car)
    
    # Simulating image input (use actual camera feed in Duckietown)
    dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)  # Placeholder image for testing

    # Run the update method with dummy image
    driver.update(dummy_image)
