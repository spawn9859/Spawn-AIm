import numba
import numpy as np
from numba import cuda

@numba.njit(parallel=True, fastmath=True, cache=True)
def parallel_preprocess_frame(np_frame: np.ndarray) -> np.ndarray:
    """Optimized frame preprocessing using Numba with parallel processing and fast math"""
    if np_frame.shape[2] == 4:
        return np_frame[:, :, :3].copy()  # Ensure contiguous memory
    return np_frame.copy()

@cuda.jit
def gpu_calculate_targets(boxes, width, height, headshot_percent, targets, distances):
    """CUDA kernel for target calculation"""
    idx = cuda.grid(1)
    if idx < boxes.shape[0]:
        width_half = width / 2
        height_half = height / 2
        
        # Minimize memory access by storing calculations
        box_x = (boxes[idx, 0] + boxes[idx, 2]) / 2
        box_y = (boxes[idx, 1] + boxes[idx, 3]) / 2
        
        x = box_x - width_half
        y = (box_y + headshot_percent * (boxes[idx, 1] - box_y)) - height_half
        
        targets[idx, 0] = x
        targets[idx, 1] = y
        distances[idx] = cuda.sqrt(x * x + y * y)

@numba.njit(parallel=True, fastmath=True, cache=True)
def cpu_calculate_targets(boxes: np.ndarray, width: float, height: float, headshot_percent: float) -> tuple:
    """Optimized CPU fallback for target calculation"""
    width_half = width / 2
    height_half = height / 2
    
    x = ((boxes[:, 0] + boxes[:, 2]) / 2) - width_half
    y = ((boxes[:, 1] + boxes[:, 3]) / 2 + 
         headshot_percent * (boxes[:, 1] - ((boxes[:, 1] + boxes[:, 3]) / 2))) - height_half
    
    targets = np.column_stack((x, y))
    distances = np.sqrt(np.sum(targets * targets, axis=1))
    
    return targets, distances

