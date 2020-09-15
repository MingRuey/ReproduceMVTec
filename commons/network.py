"""
Implemnet the T, T' described in 3.1 of the paper
"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import MaxPool2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

def _conv2d(inputs, n_filter: int, ksize: int):
    conv2d = keras.layers.Conv2D(
        filters=n_filter, kernel_size=(ksize, ksize),
        strides=(1, 1), padding="valid", dilation_rate=(1, 1),
        groups=1, activation=None, use_bias=True,
        kernel_initializer="glorot_uniform"
    )(inputs)
    activate = LeakyReLU(alpha=5e-3)(conv2d)
    return activate


def spawn_network(patch_size: int):
    """Get a T' decribed in papaer for various input size

    Args:
        patch_size (int):
            the input image patch size,
            support only 17, 33, 65

    Returns:
        keras.Model of corresponding architecture
    """
    name = "tprime_psize{}".format(patch_size)
    inputs = keras.layers.Input(shape=(patch_size, patch_size, 3))
    if (patch_size == 17):
        conv1 = _conv2d(inputs, n_filter=128, ksize=5)
        conv2 = _conv2d(conv1, n_filter=256, ksize=5)
        conv3 = _conv2d(conv2, n_filter=256, ksize=5)
        conv4 = _conv2d(conv3, n_filter=128, ksize=5)
        decode = _conv2d(conv4, n_filter=512, ksize=1)
    elif (patch_size == 33):
        conv1 = _conv2d(inputs, n_filter=128, ksize=3)
        maxp1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv1)
        conv2 = _conv2d(maxp1, n_filter=256, ksize=5)
        maxp2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv2)
        conv3 = _conv2d(maxp2, n_filter=256, ksize=2)
        conv4 = _conv2d(conv3, n_filter=128, ksize=4)
        decode = _conv2d(conv4, n_filter=512, ksize=1)
    elif (patch_size==65):
        conv1 = _conv2d(inputs, n_filter=128, ksize=5)
        maxp1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv1)
        conv2 = _conv2d(maxp1, n_filter=128, ksize=5)
        maxp2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv2)
        conv3 = _conv2d(maxp2, n_filter=128, ksize=5)
        maxp3 = MaxPool2D(pool_size=(2, 2), strides=2)(conv3)
        conv4 = _conv2d(maxp3, n_filter=256, ksize=4)
        conv5 = _conv2d(conv4, n_filter=128, ksize=1)
        decode = _conv2d(conv5, n_filter=512, ksize=1)
    else:
        raise ValueError("Unspported size: {}".format(patch_size))
    return keras.Model(inputs=inputs, outputs=decode, name=name)
