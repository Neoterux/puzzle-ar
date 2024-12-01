import cv2
from cv2.typing import MatLike
from numpy import ndarray
from typing import Union


def load_image(src: Union[str, ndarray, MatLike]) -> MatLike:
    if src is str:
        output = cv2.imread(str)
    else:
        output = src
    return output
