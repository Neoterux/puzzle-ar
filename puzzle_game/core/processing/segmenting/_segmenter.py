from abc import ABC, abstractmethod
from numpy import ndarray
from cv2.typing import MatLike
from puzzle_game.core.processing.models import Segment
from typing import Union, List


class ImageSegmenter(ABC):
    """
    Segmentador de imagen
    se encarga de dividir una imagen por las dimensiones dadas
    """

    @abstractmethod
    def segment_image(
        self, img: Union[str, ndarray, MatLike], rows: int, cols: int
    ) -> List[Segment]:
        """
        Divide la imagen en una matriz de segmentos.

        :param img: imagen de entrada
        :param rows: número de filas en la matriz
        :param cols: Número de columnas en la matriz
        :return: Lista de segmentos de la imagen
        """
        pass

    def shuffle_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Mezcla de forma aleatoria los segmentos de la imagen.
        :return: Lista de segmentos mezclados
        """
        pass
