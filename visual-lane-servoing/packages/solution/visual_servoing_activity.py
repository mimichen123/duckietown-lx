from typing import Tuple
import numpy as np
import cv2

# Define the color parameters (R, G, B, tolerance)
yellow = (254, 240, 82, 35)
white  = (254, 255, 253, 40)

# Tolerances for color detection
YELLOW_TOLERANCE = 75
WHITE_TOLERANCE  = 40

# Lane‐width constant (pixels)
LANE_WIDTH_PX = 240

def is_color(pixel: np.ndarray, color: Tuple[int,int,int,int], tol: int) -> bool:
    """Return True if `pixel` is within `tol` of `color`."""
    return (abs(int(pixel[0]) - color[0]) < tol and
            abs(int(pixel[1]) - color[1]) < tol and
            abs(int(pixel[2]) - color[2]) < tol)

def analyze_img(image: np.ndarray) -> Tuple[Tuple[int,int], Tuple[int,int], None]:
    """
    Rough scan for one yellow and one white pixel (step=4).
    Returns (yellow_px, white_px, None).
    """
    yellow_px = None
    white_px  = None

    for i in range(0, image.shape[0], 4):
        for j in range(0, image.shape[1], 4):
            pix = image[i, j]
            if yellow_px is None and is_color(pix, yellow, YELLOW_TOLERANCE):
                yellow_px = (i, j)
            if white_px is None and is_color(pix, white, WHITE_TOLERANCE):
                white_px = (i, j)
            if yellow_px and white_px:
                break
        if yellow_px and white_px:
            break

    return yellow_px, white_px, None

def get_steer_matrix_left_lane_markings(shape: Tuple[int,int]) -> np.ndarray:
    rows, cols = shape
    M = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = - (j / cols)
    return M

def get_steer_matrix_right_lane_markings(shape: Tuple[int,int]) -> np.ndarray:
    rows, cols = shape
    M = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = j / cols
    return M

def detect_lane_markings(image: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Returns (mask_left_edge, mask_right_edge, lane_center).
    - mask_left_edge: 255 where yellow line is detected
    - mask_right_edge:255 where white line is detected
    - lane_center:    (row, col) float center between detected lines
    """
    # 1) Convert to HSV and threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # these are your HSV bounds for yellow and white:
    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
    white_lower  = np.array([  0,   0, 200], dtype=np.uint8)
    white_upper  = np.array([180,  25, 255], dtype=np.uint8)

    mask_left_edge  = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_right_edge = cv2.inRange(hsv, white_lower,  white_upper)

     # 2) pick a representative pixel from each mask
    pts_y = cv2.findNonZero(mask_left_edge)
    if pts_y is not None:
        # pts_y is Nx1x2 array of (x,y) coords, take the first one
        x, y = pts_y[0][0]
        yellow_px = (int(y), int(x))
    else:
        yellow_px = None

    pts_w = cv2.findNonZero(mask_right_edge)
    if pts_w is not None:
        x, y = pts_w[0][0]
        white_px = (int(y), int(x))
    else:
        white_px = None

    # 3) Compute lane center
    if yellow_px and white_px:
        lane_center = ((yellow_px[0] + white_px[0]) / 2.0,
                       (yellow_px[1] + white_px[1]) / 2.0)
    elif white_px:
        lane_center = (white_px[0], white_px[1] - LANE_WIDTH_PX/2.0)
    elif yellow_px:
        lane_center = (yellow_px[0], yellow_px[1] + LANE_WIDTH_PX/2.0)
    else:
        lane_center = (0.0, image.shape[1]/2.0)

    return mask_left_edge, mask_right_edge, lane_center

def compute_steering(image: np.ndarray) -> float:
    mask_left, mask_right, _ = detect_lane_markings(image)
    ML = get_steer_matrix_left_lane_markings(mask_left.shape)
    MR = get_steer_matrix_right_lane_markings(mask_right.shape)
    return float(np.sum(ML * mask_left) + np.sum(MR * mask_right))
