import cv2
import numpy as np

class ImageSegmenter:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize image to be compatible with 8x8 grid
        height, width = self.original_image.shape[:2]
        new_size = (width if width > height else height)
        self.processed_image = cv2.resize(self.original_image, (new_size, new_size))
        
    def get_segments(self):
        """
        Divide the image into 8x8 segments
        Returns a list of segments as numpy arrays
        """
        height, width = self.processed_image.shape[:2]
        cell_height = height // 8
        cell_width = width // 8
        segments = []
        
        for i in range(8):
            for j in range(8):
                y1 = i * cell_height
                y2 = (i + 1) * cell_height
                x1 = j * cell_width
                x2 = (j + 1) * cell_width
                segment = self.processed_image[y1:y2, x1:x2]
                segments.append(segment)
        
        return segments
    
    def get_segment_size(self):
        """Returns the size of each segment (height, width)"""
        height, width = self.processed_image.shape[:2]
        return height // 8, width // 8
