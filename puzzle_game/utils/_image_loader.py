import cv2
from cv2.typing import MatLike
from numpy import ndarray
from typing import Union


def load_image(src: Union[str, ndarray, MatLike]) -> MatLike:
    if isinstance(src, str):
        return cv2.imread(src, cv2.IMREAD_COLOR)
    else:
        return src
