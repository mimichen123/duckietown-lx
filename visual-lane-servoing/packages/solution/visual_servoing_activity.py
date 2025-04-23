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

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Returns (mask_left_edge, mask_right_edge, lane_center).
    - mask_left_edge: 255 where yellow line is detected
    - mask_right_edge:255 where white line is detected
    - lane_center:    (row, col) float center between detected lines
    """
    # blank masks
    mask_left_edge  = np.zeros_like(image[:, :, 0])
    mask_right_edge = np.zeros_like(image[:, :, 0])

    # pixel‐wise color thresholding
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            pix = image[r, c]
            if is_color(pix, yellow, YELLOW_TOLERANCE):
                mask_left_edge[r, c] = 255
            if is_color(pix, white, WHITE_TOLERANCE):
                mask_right_edge[r, c] = 255

    # scan for one representative pixel of each
    yellow_px, white_px, _ = analyze_img(image)

    # compute center
    if yellow_px and white_px:
        lane_center = ((yellow_px[0] + white_px[0]) / 2.0,
                       (yellow_px[1] + white_px[1]) / 2.0)
    elif white_px and not yellow_px:
        lane_center = (white_px[0], white_px[1] - LANE_WIDTH_PX/2.0)
    elif yellow_px and not white_px:
        lane_center = (yellow_px[0], yellow_px[1] + LANE_WIDTH_PX/2.0)
    else:
        lane_center = (0.0, image.shape[1]/2.0)

    return mask_left_edge, mask_right_edge

def compute_steering(image: np.ndarray) -> float:
    """
    Returns a single steering command (sum of weighted masks).
    """
    mask_left, mask_right, _ = detect_lane_markings(image)
    ML = get_steer_matrix_left_lane_markings(mask_left.shape)
    MR = get_steer_matrix_right_lane_markings(mask_right.shape)
    return float(np.sum(ML * mask_left) + np.sum(MR * mask_right))
