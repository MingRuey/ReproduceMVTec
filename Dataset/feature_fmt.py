"""
The feature and parser format for MVTecAD dataset
"""

from pathlib import Path
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
        mask = tf.image.decode_image(mask, channels=0, expand_animations=False)
        if self._shp is not None:
            mask = tf.image.resize(
                mask, self._shp,
                tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
        return {"mask": mask[..., 0]}

    def _create_from(self, mask):
        mask = Path(mask)
        if not mask.is_file():
            raise ValueError("Invalid mask image path: {}".format(mask))

        with open(str(mask), 'rb') as f:
            mask = f.read()

        return {"mask": Feature._tffeature_bytes(mask)}


class MVTectFMT(ParserFMT):
    features = [Feature.ImageFeature(channels=3), SegmentationMask(n_class=2)]
