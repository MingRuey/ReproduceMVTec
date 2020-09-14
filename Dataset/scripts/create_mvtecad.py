"""
Script for creating MVTecAD dataset:
https://www.mvtec.com/company/research/datasets/mvtec-ad/
"""

import logging
from operator import gt
from pathlib import Path
from numpy.lib.npyio import load

from tensorflow.python import train

from MLBOX.Database import DBuilder, Feature
from MLBOX.Database import DBLoader
from Dataset.feature_fmt import MVTectFMT


CLASSES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper"
}

RAWDATA_DIR = Path("/rawdata/MVTecAD")
DST_DIR = "/rawdata2/TFDataset/MVTecAD"


def train_data_gener():
    for cate in CLASSES:
        dirpath = RAWDATA_DIR.joinpath(cate).joinpath("train").joinpath("good")
        for file in dirpath.rglob("*.*"):
            yield { "image": str(file), "label": [str(cate), "good"]}


def test_data_gener():
    for cate in CLASSES:
        dirpath = RAWDATA_DIR.joinpath(cate).joinpath("test")
        gtpath = RAWDATA_DIR.joinpath(cate).joinpath("ground_truth")
        for file in dirpath.rglob("*.*"):
            broken_type = file.parent.name
            features = {
                "image": str(file),
                "label":[str(cate), str(broken_type)],
            }
            if broken_type != "good":
                mask_file = gtpath.joinpath(broken_type).joinpath(file.stem + "_mask.png")
                assert mask_file.is_file(), mask_file
                features["mask"] = str(mask_file)

            yield features


def build_mvtecad():
    parser = MVTectFMT()
    builder = DBuilder(name="MVTectAD", parser=parser)

    builder.build_tfrecords(
        generator=train_data_gener(),
        output_dir=DST_DIR, split="train",
        num_of_tfrecords=20
    )

    builder.build_tfrecords(
        generator=test_data_gener(),
        output_dir=DST_DIR, split="test",
        num_of_tfrecords=10
    )


def load_mvtectad():
    parser = MVTectFMT()
    loader = DBLoader()

    loader.load(DST_DIR, parser=parser)
    train = loader.train.to_tfdataset(1, 1)
    for example in train:
        print(example)
        break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    build_mvtecad()
    # load_mvtectad()
