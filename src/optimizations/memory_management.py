import contextlib
from typing import Generator
import numpy as np

class FramePool:
    def __init__(self, max_size: int = 10):
        self.pool = []
        self.max_size = max_size

    @contextlib.contextmanager
    def get_frame(self, shape: tuple) -> Generator[np.ndarray, None, None]:
        if self.pool:
            frame = self.pool.pop()
        else:
            frame = np.empty(shape, dtype=np.uint8)
        try:
            yield frame
        finally:
            if len(self.pool) < self.max_size:
                self.pool.append(frame)

class CacheManager:
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        
    def get_cached_result(self, key: tuple) -> tuple:
        return self.cache.get(key)
        
    def cache_result(self, key: tuple, value: tuple) -> None:
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
