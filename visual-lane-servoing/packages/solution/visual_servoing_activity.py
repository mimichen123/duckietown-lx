from typing import Tuple
import numpy as np
import cv2

# HSV bounds for yellow
YELLOW_LOWER = np.array([20, 100, 100], dtype=np.uint8)
YELLOW_UPPER = np.array([30, 255, 255], dtype=np.uint8)

# HSV bounds for white (tuned to reject window glare)
WHITE_LOWER = np.array([0, 0, 215], dtype=np.uint8)
WHITE_UPPER = np.array([180, 50, 255], dtype=np.uint8)

LANE_WIDTH_PX = 240  # used if only one line found

def detect_lane_markings(image: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # yellow same as before
    mask_left = cv2.inRange(hsv,
        np.array([20, 100, 100], dtype=np.uint8),
        np.array([30, 255, 255], dtype=np.uint8)
    )

    # tighten white: very low saturation, high value
    white_lower = np.array([0, 0, 180], dtype=np.uint8)
    white_upper = np.array([180, 30, 255], dtype=np.uint8)

    # bottom 40% ROI
    y0 = int(h * 0.6)
    roi = hsv[y0:, :]

    # initial mask, open then close
    mask_roi = cv2.inRange(roi, white_lower, white_upper)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN,  kernel)
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)

    # put it back
    mask_right = np.zeros((h, w), dtype=np.uint8)
    mask_right[y0:,:] = mask_roi

    # sample one point from each
    def sample(mask):
        pts = cv2.findNonZero(mask)
        if pts is None: return None
        x, y = pts[0][0]
        return (int(y), int(x))

    yellow_px = sample(mask_left)
    white_px  = sample(mask_right)

    # compute lane center
    if yellow_px and white_px:
        lane_center = ((yellow_px[0] + white_px[0]) / 2.0,
                       (yellow_px[1] + white_px[1]) / 2.0)
    elif white_px:
        lane_center = (white_px[0], white_px[1] - LANE_WIDTH_PX/2.0)
    elif yellow_px:
        lane_center = (yellow_px[0], yellow_px[1] + LANE_WIDTH_PX/2.0)
    else:
        lane_center = (0.0, w/2.0)

    return mask_left, mask_right, lane_center

def get_steer_matrix_left(shape: Tuple[int,int]) -> np.ndarray:
    rows, cols = shape
    M = np.zeros((rows, cols), float)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = - j/cols
    return M

def get_steer_matrix_right(shape: Tuple[int,int]) -> np.ndarray:
    rows, cols = shape
    M = np.zeros((rows, cols), float)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = j/cols
    return M

def compute_steering(image: np.ndarray) -> float:
    mask_left, mask_right, _ = detect_lane_markings(image)
    ML = get_steer_matrix_left(mask_left.shape)
    MR = get_steer_matrix_right(mask_right.shape)
    return float(np.sum(ML * mask_left) + np.sum(MR * mask_right))
