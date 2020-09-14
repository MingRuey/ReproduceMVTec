import numpy as np
import pytest
import tensorflow as tf

from Dataset.unittests.configs import SAMPLE_FILES_DIR
from Dataset.feature_fmt import MVTectFMT, SegmentationMask


def encode_decode(feat, **kwargs):
    encoded = feat._create_from(**kwargs)
    example = tf.train.Example(
        features=tf.train.Features(feature=encoded)
    )
    parsed = tf.io.parse_single_example(
        example.SerializeToString(), features=feat.encoded_features
    )
    parsed = feat._parse_from(**parsed)
    return encoded, parsed


class TestSegmentationFeature:

    sample = str(SAMPLE_FILES_DIR.joinpath("000_mask.png"))

    @pytest.mark.parametrize("resize", [False, True])
    def test_encode_decode_in_eager(self, resize):
        if resize:
            out_shape = (256, 256)
            feat = SegmentationMask(resize_shape=out_shape, n_class=2)
        else:
            out_shape = (900, 900)
            feat = SegmentationMask(n_class=2)

        encoded, decoded = encode_decode(feat, mask=TestSegmentationFeature.sample)
        assert encoded.keys() == feat.encoded_features.keys()

        mask = decoded["mask"]
        assert mask.shape == out_shape
        assert mask.dtype == tf.uint8
        values, _ = tf.unique(tf.reshape(mask, [-1]))
        assert set(values.numpy()) == {0, 255}

    @pytest.mark.parametrize("resize", [False, True])
    def test_encode_decode_in_graph(self, resize):
        if resize:
            out_shape = (256, 256)
            feat = SegmentationMask(resize_shape=out_shape, n_class=2)
        else:
            out_shape = (900, 900)
            feat = SegmentationMask(n_class=2)

        @tf.function
        def encode_decode(feat, mask):
            encoded = feat._create_from(mask=mask)
            example = tf.train.Example(
                features=tf.train.Features(feature=encoded)
            )
            parsed = tf.io.parse_single_example(
                example.SerializeToString(), features=feat.encoded_features
            )
            parsed = feat._parse_from(**parsed)
            return parsed

        decoded = encode_decode(feat, mask=TestSegmentationFeature.sample)
        mask = decoded["mask"]
        assert mask.shape == out_shape
        assert mask.dtype == tf.uint8
        values, _ = tf.unique(tf.reshape(mask, [-1]))
        assert set(values.numpy()) == {0, 255}


class TestMVTectFormat:

    mask = str(SAMPLE_FILES_DIR.joinpath("000_mask.png"))
    image = str(SAMPLE_FILES_DIR.joinpath("000_target.png"))

    def test_create_and_parse_image_with_mask(self):
        fmt = MVTectFMT()
        features = {"image": str(self.image), "mask": str(self.mask), "label": "big defect" }

        example = fmt.to_example(**features)
        outputs = fmt.parse_example(example.SerializeToString())

        assert outputs["classes"] == ["big defect"]
        assert outputs["image_content"].shape == (900, 900, 3)
        mask = outputs["mask"]
        assert mask.shape == (900, 900)
        assert np.any(mask.numpy())

    def test_create_and_parse_image_wo_mask(self):
        fmt = MVTectFMT()
        features = {"image": str(self.image), "label": "big defect" }

        example = fmt.to_example(**features)
        outputs = fmt.parse_example(example.SerializeToString())

        assert outputs["classes"] == ["big defect"]
        assert outputs["image_content"].shape == (900, 900, 3)
        mask = outputs["mask"]
        assert mask.shape == (900, 900)
        assert not np.any(mask.numpy())
