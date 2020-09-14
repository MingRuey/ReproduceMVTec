"""
The feature and parser format for MVTecAD dataset
"""

from pathlib import Path
from typing import Dict, Tuple
import tensorflow as tf

from MLBOX.Database import Feature, ParserFMT


class SegmentationMask(Feature.Feature):
    encoded_features = {
        'mask': tf.io.FixedLenFeature([], tf.string, default_value="")
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
        if tf.strings.length(mask) <= 0:
            return {"mask": tf.zeros((0,), dtype=tf.uint8)}

        mask = tf.image.decode_image(mask, channels=0, expand_animations=False)
        if self._shp is not None:
            mask = tf.image.resize(
                mask, self._shp,
                tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
        return {"mask": mask[..., 0]}

    def _create_from(self, mask=""):
        if not mask:
            return dict()

        mask = Path(mask)
        if not mask.is_file():
            raise ValueError("Invalid mask image path: {}".format(mask))

        with open(str(mask), 'rb') as f:
            mask = f.read()

        return {"mask": Feature._tffeature_bytes(mask)}


class MVTectFMT(ParserFMT):
    features = [
        Feature.ImageFeature(channels=3),
        SegmentationMask(n_class=2),
        Feature.StrLabel()
    ]

    def parse_example(self, example: tf.Tensor) -> Dict[str, tf.Tensor]:
        outputs = super().parse_example(example)
        if tf.size(outputs["mask"]) <= 0:
            outputs["mask"] = tf.zeros(tf.shape(outputs["image_content"])[:2], dtype=tf.uint8)
        return outputs
