import numba
import numpy as np
from numba import cuda
import cv2

@numba.njit(parallel=True)
def parallel_preprocess_frame(np_frame: np.ndarray) -> np.ndarray:
    """
    Optimized frame preprocessing using Numba parallel processing
    """
    if np_frame.shape[2] == 4:
        return np_frame[:, :, :3]
    return np_frame

@cuda.jit
def gpu_calculate_targets(boxes, width, height, headshot_percent, targets, distances):
    """
    CUDA-accelerated target calculation
    """
    idx = cuda.grid(1)
    if idx < boxes.shape[0]:
        width_half = width / 2
        height_half = height / 2
        
        x = ((boxes[idx, 0] + boxes[idx, 2]) / 2) - width_half
        y = ((boxes[idx, 1] + boxes[idx, 3]) / 2 + 
             headshot_percent * (boxes[idx, 1] - ((boxes[idx, 1] + boxes[idx, 3]) / 2))) - height_half
        
        targets[idx, 0] = x
        targets[idx, 1] = y
        distances[idx] = np.sqrt(x * x + y * y)

