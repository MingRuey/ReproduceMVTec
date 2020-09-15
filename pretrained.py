import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from MLBOX.Database import DBLoader
from MLBOX.Database.builtin.parsers import IMAGENET
from MLBOX.Database.core.features import ImageFeature, IntLabel


def get_resnet():
    model = keras.applications.ResNet50V2(include_top=True, weights="imagenet")
    return model

class ReshapeImageNet(IMAGENET):

    features = [
        ImageFeature(resize_shape=(224, 224), channels=3),
        IntLabel(1000)
    ]

    def parse_example(self, example: tf.Tensor):
        data = super().parse_example(example)
        data["image"] = keras.applications.resnet_v2.preprocess_input(data["image"])
        return data["image"], data["label"]


def _check_acc_on_imagenet():
    loader = DBLoader()
    loader.load_built_in("imagenet", parser=ReshapeImageNet())
    imagenet = loader.test.to_tfdataset(1, 1)
    resnet50 = get_resnet()
    resnet50.compile(metrics="acc")
    resnet50.evaluate(x=imagenet)


if __name__ == "__main__":
    _check_acc_on_imagenet()
