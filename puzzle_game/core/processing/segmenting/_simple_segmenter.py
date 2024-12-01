from ._segmenter import ImageSegmenter
from puzzle_game.utils import load_image
from puzzle_game.core.processing.models import Segment, Position


class SimpleSegmenter(ImageSegmenter):

    def segment_image(self, img, rows, cols):
        image = load_image(img)
        segments = []
        imh = image.shape[0]
        imw = image.shape[1]
        M = imh // rows
        N = imw // cols

        for i in range(rows):
            for j in range(cols):
                segment_img = image[M * i : M * (i + 1), N * j : N * (j + 1)]
                original_pos = Position(x=i, y=j)

                segments.append(
                    Segment(
                        image=segment_img,
                        original_position=original_pos,
                        current_position=original_pos,
                    )
                )
        return segments

    def shuffle_segments(self, segments):
        import random

        copy = [x for x in segments]

        random.shuffle(copy)
        return copy
