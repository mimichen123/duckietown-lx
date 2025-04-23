from typing import Tuple
import numpy as np
import cv2

# HSV bounds for yellow
YELLOW_LOWER = np.array([20, 100, 100], dtype=np.uint8)
YELLOW_UPPER = np.array([30, 255, 255], dtype=np.uint8)

# HSV bounds for white (tuned to reject window glare)
WHITE_LOWER = np.array([0, 0, 150], dtype=np.uint8)
WHITE_UPPER = np.array([180, 60, 255], dtype=np.uint8)

LANE_WIDTH_PX = 240  # used if only one line found

def detect_lane_markings(image: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, Tuple[float,float]]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1) yellow mask over whole image
    mask_left = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    # 2) white mask only in bottom 50%
    y0 = int(h * 0.5)
    roi = hsv[y0:, :]
    mask_roi = cv2.inRange(roi, WHITE_LOWER, WHITE_UPPER)

    # brightness mask
    _, bright = cv2.threshold(roi[:,:,2], 150, 255, cv2.THRESH_BINARY)
    mask_roi = cv2.bitwise_and(mask_roi, bright)

    # morphological cleanup
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN,  kernel)
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)

    mask_right = np.zeros((h, w), dtype=np.uint8)
    mask_right[y0:,:] = mask_roi

    # 3) sample one point from each mask for centering
    def sample(mask):
        pts = cv2.findNonZero(mask)
        if pts is None:
            return None
        x, y = pts[0][0]
        return (int(y), int(x))

    yellow_px = sample(mask_left)
    white_px  = sample(mask_right)

    # 4) compute lane center
    if yellow_px and white_px:
        lane_center = ((yellow_px[0] + white_px[0]) / 2.0,
                       (yellow_px[1] + white_px[1]) / 2.0)
    elif white_px:
        lane_center = (white_px[0], white_px[1] - LANE_WIDTH_PX/2.0)
    elif yellow_px:
        lane_center = (yellow_px[0], yellow_px[1] + LANE_WIDTH_PX/2.0)
    else:
        lane_center = (0.0, w/2.0)

    return mask_left, mask_right


def get_steer_matrix_left_lane_markings(shape: Tuple[int,int]) -> np.ndarray:
    rows, cols = shape
    M = np.zeros((rows, cols), float)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = - j/cols
    return M


def get_steer_matrix_right_lane_markings(shape: Tuple[int,int]) -> np.ndarray:
    rows, cols = shape
    M = np.zeros((rows, cols), float)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = j/cols
    return M


def compute_steering(image: np.ndarray) -> float:
    mask_left, mask_right, _ = detect_lane_markings(image)
    ML = get_steer_matrix_left_lane_markings(mask_left.shape)
    MR = get_steer_matrix_right_lane_markings(mask_right.shape)
    return float(np.sum(ML * mask_left) + np.sum(MR * mask_right))
