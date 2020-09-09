"""
The input pipeline
"""
from typing import Tuple
import tensorflow as tf

from MLBOX.Database import DBuilder, Feature, ParserFMT
from MLBOX.Database import DBLoader


class SegmentationMask(Feature.Feature):
    encoded_features = {
        'mask': tf.io.FixedLenFeature([], tf.string)
    }

    def __init__(
            self,
            resize_shape: Tuple[int, int, int] = None,
            n_class: int = 2
            ):
        """Create feature for segmentation ground-truth

        Args:
            resize_shape (Tuple[int, int, int]):
                Specifying uniform resizing shape apply to mask.
                Defaults to None.
            n_class (int, optional):
                Number of class in segmentaion mask. Defaults to 2.
        """
        self._shp = resize_shape
        self._nClass = n_class

    def _parse_from(self, mask):
        img = tf.image.decode_image(
            mask, channels=self._channels,
            expand_animations=False
        )
        img = tf.cast(img, tf.float32)

        if self._shp is not None:
            img = tf.image.resize(
                img, self._shp,
                tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
        return img


class MVTectFMT(ParserFMT):
    features = [Feature.ImageFeature(channels=3)]
